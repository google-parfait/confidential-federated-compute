#![feature(test)]
#![allow(dead_code)]

extern crate test;

#[cfg(test)]
mod benchmarks {
    use cfc_crypto::*;
    use coset::CborSerializable;
    use test::Bencher;

    #[bench]
    fn bench_rewrap_symmetric_key(b: &mut Bencher) {
        // Encrypt the original message - expected to be the size of symmetric key (16
        // bytes)
        let plaintext = b"16-byte-text-012";
        // Associated data for the original message is supposed to be 60 bytes - the
        // size of BlobHeader.
        let associated_data: Vec<u8> =
            b"60-byte-associated-data-similar-in-size-to-blob-header-01234".into();
        let nonce = b"16-byte-nonce-01";
        let (private_key, public_key) = gen_keypair(b"key1");

        // Number of blobs whose symmetric keys are rewrapped in a single batch.
        let num_blobs = 1000;

        struct PerBlobInput {
            encapped_key: Vec<u8>,
            encrypted_symmetric_key: Vec<u8>,
            nonce: Vec<u8>,
        }

        // Prepare a batch of per-blob inputs.
        let mut input = Vec::<PerBlobInput>::new();

        for _ in 0..num_blobs {
            let (_ciphertext, encapped_key, encrypted_symmetric_key) =
                encrypt_message(plaintext, &public_key, &associated_data).unwrap();
            input.push(PerBlobInput { encapped_key, encrypted_symmetric_key, nonce: nonce.into() });
        }

        let public_key_bytes = public_key.to_vec().unwrap();
        let (_, recipient_public_key) = gen_keypair(b"key2");

        struct PerBlobOutput {
            encapped_key: Vec<u8>,
            encrypted_symmetric_key: Vec<u8>,
            public_key: Vec<u8>,
        }

        // The actual benchmarking code.
        b.iter(|| {
            let mut output = Vec::<PerBlobOutput>::new();
            for entry in input.iter() {
                let wrap_associated_data = [&public_key_bytes[..], &entry.nonce[..]].concat();
                let (encapped_key, encrypted_symmetric_key) = rewrap_symmetric_key(
                    &entry.encrypted_symmetric_key,
                    &entry.encapped_key,
                    &private_key,
                    &associated_data,
                    &recipient_public_key,
                    &wrap_associated_data,
                )
                .unwrap();
                output.push(PerBlobOutput {
                    encapped_key,
                    encrypted_symmetric_key,
                    public_key: public_key_bytes.to_owned(),
                });
            }
        });
    }
}
