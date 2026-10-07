// Copyright 2026 The Binius Developers

//! Reproduction: a `ZKProver` proof discloses the private witness to its verifier.
//!
//! The ZK wrapper sends every inner oracle-relation claim `<oracle, transparent> = claim` to the
//! verifier in the clear (`ZKWrappedVerifierChannel::verify_oracle_relation` reads it with a
//! plain `recv_one`). The transparent vector is computable from public data and Fiat-Shamir
//! challenges, so the claim is a known linear function of the committed trace, which is the
//! packed private words.
//!
//! The verifier below uses only public APIs. It replays the first half of `ZKVerifier::verify`
//! over the proof bytes, with two thin recording wrappers: one under the BaseFold channel that
//! logs every received field element, and one over the wrapped channel that, at each
//! `verify_oracle_relation`, notes the claim and tabulates the transparent polynomial.
//! Neither changes what the verifier reads, and each proof is first checked with
//! `ZKVerifier::verify`.

use std::{cell::RefCell, rc::Rc};

use binius_core::{
	constraint_system::{ConstraintSystem, ValueVec},
	word::Word,
};
use binius_field::{
	Field, Ghash128b as B128, arch::OptimalPackedB128, arithmetic_traits::InvertOrZero,
};
use binius_frontend::CircuitBuilder;
use binius_hash::StdHashSuite;
use binius_iop::{
	channel::{IOPVerifierChannel, OracleSpec, TransparentEvalFn, merge::MergeVerifierChannel},
	merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
};
use binius_ip::channel::{Error as ChannelError, IPVerifierChannel, WordIPVerifierChannel};
use binius_prover::zk_config::ZKProver;
use binius_spartan_frontend::circuit_builder::InstanceGenerator;
use binius_spartan_verifier::wrapper::{ZKWrappedVerifierChannel, circuit_elem::CircuitElem};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::{config::StdChallenger, zk_config::ZKVerifier};
use rand::{SeedableRng, rngs::StdRng};

type Elem = CircuitElem<B128, InstanceGenerator<B128>>;

/// Under the BaseFold channel: logs every field element the verifier receives from the prover.
struct RecordRecv<C> {
	inner: C,
	received: Rc<RefCell<Vec<B128>>>,
}

impl<C: IPVerifierChannel<B128, Elem = B128>> IPVerifierChannel<B128> for RecordRecv<C> {
	type Elem = B128;

	fn recv_one(&mut self) -> Result<B128, ChannelError> {
		let value = self.inner.recv_one()?;
		self.received.borrow_mut().push(value);
		Ok(value)
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<B128>, ChannelError> {
		let values = self.inner.recv_many(n)?;
		self.received.borrow_mut().extend_from_slice(&values);
		Ok(values)
	}

	fn recv_public_claim(&mut self) -> Result<B128, ChannelError> {
		let value = self.inner.recv_public_claim()?;
		self.received.borrow_mut().push(value);
		Ok(value)
	}

	fn sample(&mut self) -> B128 {
		self.inner.sample()
	}

	fn observe_one(&mut self, val: B128) -> B128 {
		self.inner.observe_one(val)
	}

	fn assert_zero(&mut self, val: B128) -> Result<(), ChannelError> {
		self.inner.assert_zero(val)
	}
}

impl<C: WordIPVerifierChannel<B128, Elem = B128, Word = Word>> WordIPVerifierChannel<B128>
	for RecordRecv<C>
{
	type Word = Word;

	fn observe_words(&mut self, words: &[Word]) -> Vec<Word> {
		self.inner.observe_words(words)
	}

	fn subset_sum(&mut self, elems: &[B128], word: &Word) -> B128 {
		self.inner.subset_sum(elems, word)
	}

	fn select(&mut self, elems: &[B128], word: &Word) -> B128 {
		self.inner.select(elems, word)
	}

	fn sample_bits(&mut self, bits: usize) -> Word {
		self.inner.sample_bits(bits)
	}

	fn pack_words(&mut self, words: &[Word]) -> Vec<B128> {
		self.inner.pack_words(words)
	}
}

impl<C: MerkleIPVerifierChannel<B128, Elem = B128, Word = Word>> MerkleIPVerifierChannel<B128>
	for RecordRecv<C>
{
	type Commitment = C::Commitment;

	fn recv_merkle_commitment(
		&mut self,
		leaf_size: usize,
		depth: usize,
	) -> Result<Self::Commitment, binius_iop::merkle_channel::Error> {
		self.inner.recv_merkle_commitment(leaf_size, depth)
	}

	fn recv_openings(
		&mut self,
		commitment: &Self::Commitment,
		indices: &[Word],
	) -> Result<Vec<B128>, binius_iop::merkle_channel::Error> {
		self.inner.recv_openings(commitment, indices)
	}

	fn recv_committed_vector(
		&mut self,
		commitment: &Self::Commitment,
	) -> Result<Vec<B128>, binius_iop::merkle_channel::Error> {
		self.inner.recv_committed_vector(commitment)
	}
}

/// One inner oracle relation as the verifier sees it: `<oracle, transparent> = claim`.
#[derive(Debug)]
struct Relation {
	claim: B128,
	/// The transparent polynomial's values on the hypercube, index `i` at the little-endian bits
	/// of `i`.
	transparent: Vec<B128>,
}

/// Over the wrapped channel: records each oracle relation, then forwards it unchanged.
struct RecordRelations<C> {
	inner: C,
	received: Rc<RefCell<Vec<B128>>>,
	relations: Vec<Relation>,
}

impl<C: IPVerifierChannel<B128, Elem = Elem>> IPVerifierChannel<B128> for RecordRelations<C> {
	type Elem = Elem;

	fn recv_one(&mut self) -> Result<Elem, ChannelError> {
		self.inner.recv_one()
	}

	fn recv_public_claim(&mut self) -> Result<Elem, ChannelError> {
		self.inner.recv_public_claim()
	}

	fn sample(&mut self) -> Elem {
		self.inner.sample()
	}

	fn observe_one(&mut self, val: B128) -> Elem {
		self.inner.observe_one(val)
	}

	fn assert_zero(&mut self, val: Elem) -> Result<(), ChannelError> {
		self.inner.assert_zero(val)
	}
}

impl<C: WordIPVerifierChannel<B128, Elem = Elem, Word = Word>> WordIPVerifierChannel<B128>
	for RecordRelations<C>
{
	type Word = Word;

	fn observe_words(&mut self, words: &[Word]) -> Vec<Word> {
		self.inner.observe_words(words)
	}

	fn subset_sum(&mut self, elems: &[Elem], word: &Word) -> Elem {
		self.inner.subset_sum(elems, word)
	}

	fn select(&mut self, elems: &[Elem], word: &Word) -> Elem {
		self.inner.select(elems, word)
	}

	fn sample_bits(&mut self, bits: usize) -> Word {
		self.inner.sample_bits(bits)
	}

	fn pack_words(&mut self, words: &[Word]) -> Vec<Elem> {
		self.inner.pack_words(words)
	}
}

impl<C: IOPVerifierChannel<B128, Elem = Elem>> IOPVerifierChannel<B128> for RecordRelations<C> {
	/// The handle carries the oracle's `log_msg_len`, so the relation knows its size.
	type Oracle = (C::Oracle, usize);

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		self.inner.remaining_oracle_specs()
	}

	fn recv_oracle(
		&mut self,
		len: usize,
		log_msg_len: usize,
		is_witness_dependent: bool,
	) -> Result<Self::Oracle, binius_iop::channel::Error> {
		let oracle = self
			.inner
			.recv_oracle(len, log_msg_len, is_witness_dependent)?;
		Ok((oracle, log_msg_len))
	}

	fn verify_oracle_relation(
		&mut self,
		(oracle, log_len): Self::Oracle,
		transparent: TransparentEvalFn<Elem>,
		claim: Elem,
	) -> Result<(), binius_iop::channel::Error> {
		let value = |elem: Elem| match elem {
			CircuitElem::Constant(value) => value,
			CircuitElem::Wire { wire, .. } => wire.value().expect("transparent value is public"),
		};
		let table = (0..1usize << log_len)
			.map(|i| {
				let point = (0..log_len)
					.map(|bit| Elem::Constant([B128::ZERO, B128::ONE][(i >> bit) & 1]))
					.collect::<Vec<_>>();
				value(transparent(&point))
			})
			.collect();
		// The wrapped channel's first read is the claim, sent as a plain field element.
		let n_before = self.received.borrow().len();
		self.inner
			.verify_oracle_relation(oracle, transparent, claim)?;
		self.relations.push(Relation {
			claim: self.received.borrow()[n_before],
			transparent: table,
		});
		Ok(())
	}
}

/// Replays `ZKVerifier::verify` up to the end of the inner verifier and returns the inner oracle
/// relations it saw. Uses only what the verifier holds: the setup, the public words, the proof.
fn verifier_view(
	verifier: &ZKVerifier<StdHashSuite>,
	inout: &[Word],
	proof: Vec<u8>,
) -> Vec<Relation> {
	let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof);
	let received = Rc::new(RefCell::new(Vec::new()));
	let merkle = RecordRecv {
		inner: VerifierMerkleTranscriptChannel::<_, StdChallenger, B128, StdHashSuite>::new(
			&mut transcript,
		),
		received: Rc::clone(&received),
	};
	let basefold = verifier.basefold_compiler().create_channel(merkle);
	let wrapped = ZKWrappedVerifierChannel::new(
		MergeVerifierChannel::new(basefold, verifier.oracle_schedule()),
		verifier.outer_iop_verifier(),
		verifier.outer_layout_arc(),
	)
	.unwrap();
	let mut channel = RecordRelations {
		inner: wrapped,
		received,
		relations: Vec::new(),
	};
	let inout = channel.observe_words(inout);
	// The wiring claim is checked by `ZKVerifier::verify`; this replay only needs the relations.
	let _wiring_claim = verifier
		.inner_iop_verifier()
		.verify(&inout, &mut channel)
		.unwrap();
	channel.relations
}

/// The committed trace: the non-public words, two to a field element, zero-padded.
fn pack_trace(words: &[Word], log_len: usize) -> Vec<B128> {
	let mut trace = words
		.chunks(2)
		.map(|pair| {
			let hi = pair.get(1).map_or(0, |w| w.0);
			B128::new(((hi as u128) << 64) | pair[0].0 as u128)
		})
		.collect::<Vec<_>>();
	trace.resize(1 << log_len, B128::ZERO);
	trace
}

fn inner_product(a: &[B128], b: &[B128]) -> B128 {
	a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// `z_i = a_i XOR b_i` for each pair: `z` is public, every `a_i` and `b_i` is private.
fn xor_split(z: &[u64], a: &[u64]) -> (ConstraintSystem, ValueVec) {
	let builder = CircuitBuilder::new();
	let wires = z
		.iter()
		.map(|_| {
			let (wa, wb, wz) = (builder.add_witness(), builder.add_witness(), builder.add_inout());
			builder.assert_eq("xor", builder.bxor(wa, wb), wz);
			(wa, wb, wz)
		})
		.collect::<Vec<_>>();
	let circuit = builder.build();
	let mut filler = circuit.new_witness_filler();
	for (&(wa, wb, wz), (&z, &a)) in wires.iter().zip(z.iter().zip(a)) {
		filler[wa] = Word(a);
		filler[wb] = Word(a ^ z);
		filler[wz] = Word(z);
	}
	circuit.populate_wire_witness(&mut filler).unwrap();
	(circuit.constraint_system().clone(), filler.into_value_vec())
}

/// Proves with `ZKProver`, checks the proof with `ZKVerifier`, and returns the proof bytes.
fn prove(verifier: &ZKVerifier<StdHashSuite>, witness: &ValueVec, seed: u64) -> Vec<u8> {
	let prover = ZKProver::<OptimalPackedB128, StdHashSuite>::setup(verifier).unwrap();
	let mut transcript = ProverTranscript::new(StdChallenger::default());
	prover
		.prove(witness, StdRng::seed_from_u64(seed), &mut transcript)
		.unwrap();
	let proof = transcript.finalize();
	let mut check = VerifierTranscript::new(StdChallenger::default(), proof.clone());
	verifier.verify(witness.inout(), &mut check).unwrap();
	check.finalize().unwrap();
	proof
}

/// One XOR split, one trace element: the verifier recovers both private words from any proof.
#[test]
fn zk_proof_reveals_private_words() {
	let z = 0x5a5a_c3c3_0f0f_9696;
	let a = 0x0123_4567_89ab_cdef;
	let (cs, witness) = xor_split(&[z], &[a]);
	let verifier = ZKVerifier::<StdHashSuite>::setup(cs, 1).unwrap();
	assert_eq!(verifier.inner_iop_verifier().log_witness_elems(), 0);

	for seed in 0..4 {
		let proof = prove(&verifier, &witness, seed);
		let relations = verifier_view(&verifier, witness.inout(), proof);
		assert_eq!(relations.len(), 1);
		let Relation { claim, transparent } = &relations[0];

		// `claim = trace[0] * transparent[0]`, so the trace is the claim over the transparent.
		let trace = u128::from(*claim * transparent[0].invert_or_zero());
		assert_eq!(trace as u64, a, "recovered a");
		assert_eq!((trace >> 64) as u64, a ^ z, "recovered b");
	}
}

/// Eight XOR splits, eight trace elements, two witnesses for one statement: the claim tells the
/// verifier which witness each proof was made from.
#[test]
fn zk_proof_distinguishes_witnesses() {
	let z = [1, 2, 3, 4, 5, 6, 7, 8].map(|i: u64| i.wrapping_mul(0x9e37_79b9_7f4a_7c15));
	let a0 = [11, 12, 13, 14, 15, 16, 17, 18].map(|i: u64| i.wrapping_mul(0xbf58_476d_1ce4_e5b9));
	let a1 = a0.map(|a| a ^ 0xffff_0000_ffff_0000);
	let (cs, w0) = xor_split(&z, &a0);
	let (_, w1) = xor_split(&z, &a1);
	assert_eq!(w0.inout(), w1.inout(), "same public statement");
	assert_ne!(w0.non_public(), w1.non_public(), "different witnesses");

	let verifier = ZKVerifier::<StdHashSuite>::setup(cs, 1).unwrap();
	let log_len = verifier.inner_iop_verifier().log_witness_elems();
	assert_eq!(log_len, 3, "16 private words, two to a trace element");
	let candidates = [w0.non_public(), w1.non_public()].map(|words| pack_trace(words, log_len));

	for (seed, secret) in [0, 1, 1, 0, 1, 0, 0, 1].into_iter().enumerate() {
		let proof = prove(&verifier, [&w0, &w1][secret], seed as u64);
		let relations = verifier_view(&verifier, w0.inout(), proof);
		let Relation { claim, transparent } = &relations[0];
		assert_eq!(transparent.len(), 1 << log_len);
		let matches = candidates
			.iter()
			.map(|trace| inner_product(trace, transparent) == *claim)
			.collect::<Vec<_>>();
		let mut expected = [false, false];
		expected[secret] = true;
		assert_eq!(matches, expected, "proof {seed} identifies its witness");
	}
}
