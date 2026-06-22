// Copyright 2026 The Trusted Computations Platform Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

extern crate prost;
extern crate tcp_integration;
extern crate tcp_proto;
extern crate willow_reputable_decryptor_service;

mod test {
    use prost::bytes::Bytes;
    use prost::Message;
    use tcp_integration::harness::*;
    use tcp_proto::runtime::endpoint::out_message;
    use willow_reputable_decryptor_service::actor::ReputableDecryptorActor;
    use willow_reputable_decryptor_service::secure_aggregation::willow::ShellAhePartialDecCiphertext as ProstShellAhePartialDecCiphertext;
    use willow_reputable_decryptor_service::secure_aggregation::willow::{
        reputable_decryptor_request, reputable_decryptor_response, CreateSetupContributionRequest,
        HandlePartialDecryptionRequest, KeyContribution, PartialDecryptionRequest,
        ReputableDecryptorRequest, ReputableDecryptorResponse,
        VerifyAndAggregateKeyContributionsRequest, VerifyKeyContributionsRequest,
    };

    // Cryptographic top level test imports
    use ahe_traits::AheBase;
    use prng_traits::SecurePrng;
    use proto_serialization_traits::{FromProto, ToProto};
    use shell_parameters::create_shell_ahe_config;
    use shell_vahe::ShellVahe;
    use vahe_traits::VerifiableEncrypt;
    use willow_reputable_decryptor_service::decryptor::{convert_from_prost, convert_to_prost};

    const DEFAULT_MAX_NUMBER_OF_DECRYPTORS: usize = 1;
    const DEFAULT_MAX_NUMBER_OF_KEYS: usize = 100;

    fn advance_until_response(
        cluster: &mut FakeCluster<ReputableDecryptorActor>,
    ) -> ReputableDecryptorResponse {
        let mut reputable_decryptor_response: Option<ReputableDecryptorResponse> = None;
        let response_messages =
            cluster.advance_until(&mut |envelope_out| match &envelope_out.msg {
                Some(out_message::Msg::DeliverAppMessage(message)) => {
                    let response =
                        ReputableDecryptorResponse::decode(message.message_header.as_ref())
                            .unwrap();
                    reputable_decryptor_response = Some(response);
                    return true;
                }
                _ => false,
            });

        assert!(!response_messages.is_empty());
        reputable_decryptor_response.unwrap()
    }

    #[test]
    fn test_key_setup_and_aggregation_lifecycle() {
        let mut cluster = FakeCluster::new(Bytes::new());
        cluster.start_node(
            1,
            true,
            ReputableDecryptorActor::new(
                DEFAULT_MAX_NUMBER_OF_DECRYPTORS,
                DEFAULT_MAX_NUMBER_OF_KEYS,
            ),
        );
        cluster.advance_until_elected_leader(None);
        assert_eq!(cluster.leader_id(), 1);

        // 1. Create setup contribution for key_1
        let setup_req = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::CreateSetupContribution(
                CreateSetupContributionRequest { key_id: "test_key_1".to_string() },
            )),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            1,
            setup_req.encode_to_vec().into(),
            Bytes::new(),
        );

        let response = advance_until_response(&mut cluster);
        let c1 = match response.msg {
            Some(reputable_decryptor_response::Msg::CreateSetupContribution(resp)) => {
                let setup_contrib = resp.setup_contribution.unwrap();
                let key_contrib = setup_contrib.key_contribution.as_ref().unwrap();
                assert!(key_contrib.public_key_share.is_some());
                assert!(key_contrib.proof.is_some());
                setup_contrib
            }
            _ => panic!("Expected CreateSetupContributionResponse, got {:?}", response),
        };

        // 2. Request key_1 again, verify setup contribution is idempotent and identical
        cluster.send_app_message(
            cluster.leader_id(),
            2,
            setup_req.encode_to_vec().into(),
            Bytes::new(),
        );
        let response2 = advance_until_response(&mut cluster);
        let c2 = match response2.msg {
            Some(reputable_decryptor_response::Msg::CreateSetupContribution(resp)) => {
                resp.setup_contribution.unwrap()
            }
            _ => panic!("Expected CreateSetupContributionResponse"),
        };

        assert_eq!(
            c1.key_contribution.as_ref().unwrap().public_key_share.as_ref().unwrap().poly,
            c2.key_contribution.as_ref().unwrap().public_key_share.as_ref().unwrap().poly,
            "Setup contribution must be idempotent and return identical public key share"
        );

        // 3. Verify and aggregate public key contribution
        let mut kc1 = KeyContribution::default();
        let key_contrib = c1.key_contribution.as_ref().unwrap();
        kc1.public_key_share = Some(key_contrib.public_key_share.as_ref().unwrap().clone());
        kc1.proof = Some(key_contrib.proof.as_ref().unwrap().clone());

        let mut verify_key_contributions_request = VerifyKeyContributionsRequest::default();
        verify_key_contributions_request.key_contributions.push(kc1);

        let mut verify_req = VerifyAndAggregateKeyContributionsRequest::default();
        verify_req.key_id = "test_key_1".to_string();
        verify_req.verify_key_contributions_request = Some(verify_key_contributions_request);

        let aggregate_request = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::VerifyAndAggregateKeyContributions(
                verify_req,
            )),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            3,
            aggregate_request.encode_to_vec().into(),
            Bytes::new(),
        );
        let response3 = advance_until_response(&mut cluster);
        match response3.msg {
            Some(reputable_decryptor_response::Msg::VerifyAndAggregateKeyContributions(resp)) => {
                assert!(resp.public_key.is_some());
            }
            _ => panic!("Expected VerifyAndAggregateKeyContributionsResponse, got {:?}", response3),
        }
    }

    #[test]
    fn test_decryption_and_cache_cleanup_lifecycle() {
        let mut cluster = FakeCluster::new(Bytes::new());
        cluster.start_node(
            1,
            true,
            ReputableDecryptorActor::new(
                DEFAULT_MAX_NUMBER_OF_DECRYPTORS,
                DEFAULT_MAX_NUMBER_OF_KEYS,
            ),
        );
        cluster.advance_until_elected_leader(None);
        assert_eq!(cluster.leader_id(), 1);

        // 1. Create setup contribution for key_1
        let setup_req_1 = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::CreateSetupContribution(
                CreateSetupContributionRequest { key_id: "key_1".to_string() },
            )),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            1,
            setup_req_1.encode_to_vec().into(),
            Bytes::new(),
        );

        let resp1 = advance_until_response(&mut cluster);
        let setup_contrib = match resp1.msg {
            Some(reputable_decryptor_response::Msg::CreateSetupContribution(resp)) => {
                resp.setup_contribution.unwrap()
            }
            _ => panic!("Expected CreateSetupContributionResponse"),
        };
        let pk_1 = setup_contrib
            .key_contribution
            .as_ref()
            .unwrap()
            .public_key_share
            .as_ref()
            .unwrap()
            .clone();

        // 2. Aggregate public key contribution (single-party setup)
        let mut kc1 = KeyContribution::default();
        let key_contrib = setup_contrib.key_contribution.as_ref().unwrap();
        kc1.public_key_share = Some(key_contrib.public_key_share.as_ref().unwrap().clone());
        kc1.proof = Some(key_contrib.proof.as_ref().unwrap().clone());

        let mut verify_key_contributions_request = VerifyKeyContributionsRequest::default();
        verify_key_contributions_request.key_contributions.push(kc1);

        let mut verify_req = VerifyAndAggregateKeyContributionsRequest::default();
        verify_req.key_id = "key_1".to_string();
        verify_req.verify_key_contributions_request = Some(verify_key_contributions_request);

        let aggregate_request = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::VerifyAndAggregateKeyContributions(
                verify_req,
            )),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            2,
            aggregate_request.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp2 = advance_until_response(&mut cluster);
        let agg_pk_prost = match resp2.msg {
            Some(reputable_decryptor_response::Msg::VerifyAndAggregateKeyContributions(resp)) => {
                resp.public_key.unwrap()
            }
            _ => panic!("Expected VerifyAndAggregateKeyContributionsResponse"),
        };

        // 3. Locally encrypt valid standard plaintext under key_1's public key
        let vahe =
            ShellVahe::new(create_shell_ahe_config(1).unwrap(), b"willow_reputable_decryptor")
                .unwrap();
        let agg_pk_pb: shell_ciphertexts_rust_proto::ShellAhePublicKey =
            convert_from_prost(&agg_pk_prost).unwrap();
        let agg_pk = <ShellVahe as AheBase>::PublicKey::from_proto(agg_pk_pb, &vahe).unwrap();

        let plaintext: Vec<i64> = vec![0; 8];
        let mut prng = <ShellVahe as AheBase>::Rng::create(
            &<ShellVahe as AheBase>::Rng::generate_seed().unwrap(),
        )
        .unwrap();
        let (ciphertext, _) =
            vahe.verifiable_encrypt(&plaintext, &agg_pk, b"test_nonce", &mut prng).unwrap();

        let pd_ciphertext_pb = vahe.get_partial_dec_ciphertext(&ciphertext).unwrap();
        let pd_ciphertext_pb_proto = pd_ciphertext_pb.to_proto(&vahe).unwrap();
        let pd_ciphertext_prost: ProstShellAhePartialDecCiphertext =
            convert_to_prost(&pd_ciphertext_pb_proto).unwrap();

        // 4. Perform successful partial decryption
        let mut dec_req = HandlePartialDecryptionRequest::default();
        dec_req.key_id = "key_1".to_string();
        let mut inner_dec_req = PartialDecryptionRequest::default();
        inner_dec_req.partial_dec_ciphertext = Some(pd_ciphertext_prost.clone());
        dec_req.partial_decryption_request = Some(inner_dec_req);

        let dec_request = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::HandlePartialDecryption(dec_req)),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            3,
            dec_request.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp = advance_until_response(&mut cluster);
        match resp.msg {
            Some(reputable_decryptor_response::Msg::HandlePartialDecryption(resp)) => {
                let partial_dec = resp.partial_decryption_response.unwrap();
                assert!(partial_dec.partial_decryption.is_some());
            }
            _ => panic!("Expected HandlePartialDecryptionResponse, got {:?}", resp),
        }

        // 5. Request key_1 setup contribution again. Assert that a new public key is
        //    returned. It should be different than pk_1 because the old key was used in
        //    a previous decryption request, and the key is evicted after being used.
        cluster.send_app_message(
            cluster.leader_id(),
            4,
            setup_req_1.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp_new = advance_until_response(&mut cluster);
        let pk_3 = match resp_new.msg {
            Some(reputable_decryptor_response::Msg::CreateSetupContribution(resp)) => {
                resp.setup_contribution.unwrap().key_contribution.unwrap().public_key_share.unwrap()
            }
            _ => panic!("Expected CreateSetupContributionResponse"),
        };

        assert_ne!(
            pk_1.poly, pk_3.poly,
            "Decrypted key setup contribution was served from cache! Cache leak detected."
        );
    }

    #[test]
    fn test_eviction_lifecycle() {
        let mut cluster = FakeCluster::new(Bytes::new());
        cluster.start_node(
            1,
            true,
            ReputableDecryptorActor::new(DEFAULT_MAX_NUMBER_OF_DECRYPTORS, 1),
        );
        cluster.advance_until_elected_leader(None);

        // 1. Create setup contribution for key_1
        let setup_req_1 = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::CreateSetupContribution(
                CreateSetupContributionRequest { key_id: "key_1".to_string() },
            )),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            1,
            setup_req_1.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp1 = advance_until_response(&mut cluster);
        let setup_contrib_1 = match resp1.msg {
            Some(reputable_decryptor_response::Msg::CreateSetupContribution(resp)) => {
                resp.setup_contribution.unwrap()
            }
            _ => panic!("Expected CreateSetupContributionResponse"),
        };

        // 2. Create setup contribution for key_2, forcing eviction of key_1 (capacity
        //    limit is 1)
        let setup_req_2 = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::CreateSetupContribution(
                CreateSetupContributionRequest { key_id: "key_2".to_string() },
            )),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            2,
            setup_req_2.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp2 = advance_until_response(&mut cluster);
        let setup_contrib_2 = match resp2.msg {
            Some(reputable_decryptor_response::Msg::CreateSetupContribution(resp)) => {
                resp.setup_contribution.unwrap()
            }
            _ => panic!("Expected CreateSetupContributionResponse"),
        };

        // 3. Verify and aggregate key_1 should fail because it was evicted
        let mut kc1 = KeyContribution::default();
        let key_contrib_1 = setup_contrib_1.key_contribution.as_ref().unwrap();
        kc1.public_key_share = Some(key_contrib_1.public_key_share.as_ref().unwrap().clone());
        kc1.proof = Some(key_contrib_1.proof.as_ref().unwrap().clone());

        let mut verify_key_contributions_request_1 = VerifyKeyContributionsRequest::default();
        verify_key_contributions_request_1.key_contributions.push(kc1);

        let mut verify_req_1 = VerifyAndAggregateKeyContributionsRequest::default();
        verify_req_1.key_id = "key_1".to_string();
        verify_req_1.verify_key_contributions_request = Some(verify_key_contributions_request_1);

        let aggregate_request_1 = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::VerifyAndAggregateKeyContributions(
                verify_req_1,
            )),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            3,
            aggregate_request_1.encode_to_vec().into(),
            Bytes::new(),
        );
        let response_fail = advance_until_response(&mut cluster);
        match response_fail.msg {
            Some(reputable_decryptor_response::Msg::Error(status)) => {
                assert_eq!(status.code, 9); // FailedPrecondition
                assert!(status.message.contains("key_1 not found"));
            }
            _ => panic!("Expected Error response, got {:?}", response_fail),
        }

        // 4. Verify and aggregate key_2 should succeed
        let mut kc2 = KeyContribution::default();
        let key_contrib_2 = setup_contrib_2.key_contribution.as_ref().unwrap();
        kc2.public_key_share = Some(key_contrib_2.public_key_share.as_ref().unwrap().clone());
        kc2.proof = Some(key_contrib_2.proof.as_ref().unwrap().clone());

        let mut verify_key_contributions_request_2 = VerifyKeyContributionsRequest::default();
        verify_key_contributions_request_2.key_contributions.push(kc2);

        let mut verify_req_2 = VerifyAndAggregateKeyContributionsRequest::default();
        verify_req_2.key_id = "key_2".to_string();
        verify_req_2.verify_key_contributions_request = Some(verify_key_contributions_request_2);

        let aggregate_request_2 = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::VerifyAndAggregateKeyContributions(
                verify_req_2,
            )),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            4,
            aggregate_request_2.encode_to_vec().into(),
            Bytes::new(),
        );
        let response_success = advance_until_response(&mut cluster);
        match response_success.msg {
            Some(reputable_decryptor_response::Msg::VerifyAndAggregateKeyContributions(resp)) => {
                assert!(resp.public_key.is_some());
            }
            _ => panic!(
                "Expected VerifyAndAggregateKeyContributionsResponse, got {:?}",
                response_success
            ),
        }
    }

    #[test]
    fn test_invalid_inputs_integration() {
        let mut cluster = FakeCluster::new(Bytes::new());
        cluster.start_node(
            1,
            true,
            ReputableDecryptorActor::new(
                DEFAULT_MAX_NUMBER_OF_DECRYPTORS,
                DEFAULT_MAX_NUMBER_OF_KEYS,
            ),
        );
        cluster.advance_until_elected_leader(None);

        // 0. Initialize a valid key first to test subsequent commands
        let init_req = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::CreateSetupContribution(
                CreateSetupContributionRequest { key_id: "valid_key".to_string() },
            )),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            1,
            init_req.encode_to_vec().into(),
            Bytes::new(),
        );
        let _resp_init = advance_until_response(&mut cluster);

        // 1. CreateSetupContribution with empty key_id
        let req_1 = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::CreateSetupContribution(
                CreateSetupContributionRequest { key_id: "".to_string() },
            )),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            2,
            req_1.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp_1 = advance_until_response(&mut cluster);
        match resp_1.msg {
            Some(reputable_decryptor_response::Msg::Error(status)) => {
                assert_eq!(status.code, 3); // 3 is InvalidArgument
                assert!(status.message.contains("missing key_id field"));
            }
            _ => panic!("Expected Error response, got {:?}", resp_1),
        }

        // 2. HandlePartialDecryptionRequest with empty key_id
        let mut dec_req_2 = HandlePartialDecryptionRequest::default();
        dec_req_2.key_id = "".to_string();
        dec_req_2.partial_decryption_request = Some(PartialDecryptionRequest::default());
        let req_2 = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::HandlePartialDecryption(dec_req_2)),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            3,
            req_2.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp_2 = advance_until_response(&mut cluster);
        match resp_2.msg {
            Some(reputable_decryptor_response::Msg::Error(status)) => {
                assert_eq!(status.code, 3); // InvalidArgument
                assert!(status.message.contains("missing key_id field"));
            }
            _ => panic!("Expected Error response, got {:?}", resp_2),
        }

        // 3. HandlePartialDecryptionRequest with missing inner
        //    partial_decryption_request
        let mut dec_req_3 = HandlePartialDecryptionRequest::default();
        dec_req_3.key_id = "valid_key".to_string();
        dec_req_3.partial_decryption_request = None;
        let req_3 = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::HandlePartialDecryption(dec_req_3)),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            4,
            req_3.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp_3 = advance_until_response(&mut cluster);
        match resp_3.msg {
            Some(reputable_decryptor_response::Msg::Error(status)) => {
                assert_eq!(status.code, 3); // InvalidArgument
                assert!(status.message.contains("missing partial_decryption_request field"));
            }
            _ => panic!("Expected Error response, got {:?}", resp_3),
        }

        // 4. VerifyAndAggregateKeyContributions with empty key_id
        let mut verify_req_4 = VerifyAndAggregateKeyContributionsRequest::default();
        verify_req_4.key_id = "".to_string();
        verify_req_4.verify_key_contributions_request =
            Some(VerifyKeyContributionsRequest::default());
        let req_4 = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::VerifyAndAggregateKeyContributions(
                verify_req_4,
            )),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            5,
            req_4.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp_4 = advance_until_response(&mut cluster);
        match resp_4.msg {
            Some(reputable_decryptor_response::Msg::Error(status)) => {
                assert_eq!(status.code, 3); // InvalidArgument
                assert!(status.message.contains("missing key_id field"));
            }
            _ => panic!("Expected Error response, got {:?}", resp_4),
        }

        // 5. VerifyAndAggregateKeyContributions with missing inner
        //    verify_key_contributions_request
        let mut verify_req_5 = VerifyAndAggregateKeyContributionsRequest::default();
        verify_req_5.key_id = "some_key".to_string();
        verify_req_5.verify_key_contributions_request = None;
        let req_5 = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::VerifyAndAggregateKeyContributions(
                verify_req_5,
            )),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            6,
            req_5.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp_5 = advance_until_response(&mut cluster);
        match resp_5.msg {
            Some(reputable_decryptor_response::Msg::Error(status)) => {
                assert_eq!(status.code, 3); // InvalidArgument
                assert!(status.message.contains("missing verify_key_contributions_request field"));
            }
            _ => panic!("Expected Error response, got {:?}", resp_5),
        }

        // 6. VerifyAndAggregateKeyContributions with empty contributions list
        let mut verify_req_6 = VerifyAndAggregateKeyContributionsRequest::default();
        verify_req_6.key_id = "valid_key".to_string();
        verify_req_6.verify_key_contributions_request =
            Some(VerifyKeyContributionsRequest::default());
        let req_6 = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::VerifyAndAggregateKeyContributions(
                verify_req_6,
            )),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            7,
            req_6.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp_6 = advance_until_response(&mut cluster);
        match resp_6.msg {
            Some(reputable_decryptor_response::Msg::Error(status)) => {
                assert_eq!(status.code, 3); // InvalidArgument
                assert!(
                    status.message.contains("key_contributions list is empty"),
                    "Actual message: {}",
                    status.message
                );
            }
            _ => panic!("Expected Error response, got {:?}", resp_6),
        }
    }

    #[test]
    fn test_decryption_consumes_key_so_cannot_decrypt_twice() {
        let mut cluster = FakeCluster::new(Bytes::new());
        cluster.start_node(
            1,
            true,
            ReputableDecryptorActor::new(
                DEFAULT_MAX_NUMBER_OF_DECRYPTORS,
                DEFAULT_MAX_NUMBER_OF_KEYS,
            ),
        );
        cluster.advance_until_elected_leader(None);

        // 1. Create setup contribution for key_1
        let setup_req = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::CreateSetupContribution(
                CreateSetupContributionRequest { key_id: "key_1".to_string() },
            )),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            1,
            setup_req.encode_to_vec().into(),
            Bytes::new(),
        );

        let response = advance_until_response(&mut cluster);
        let c1 = match response.msg {
            Some(reputable_decryptor_response::Msg::CreateSetupContribution(resp)) => {
                resp.setup_contribution.unwrap()
            }
            _ => panic!("Expected CreateSetupContributionResponse, got {:?}", response),
        };

        // 2. Verify and aggregate public key contribution
        let mut kc1 = KeyContribution::default();
        let key_contrib = c1.key_contribution.as_ref().unwrap();
        kc1.public_key_share = Some(key_contrib.public_key_share.as_ref().unwrap().clone());
        kc1.proof = Some(key_contrib.proof.as_ref().unwrap().clone());

        let mut verify_key_contributions_request = VerifyKeyContributionsRequest::default();
        verify_key_contributions_request.key_contributions.push(kc1);

        let mut verify_req = VerifyAndAggregateKeyContributionsRequest::default();
        verify_req.key_id = "key_1".to_string();
        verify_req.verify_key_contributions_request = Some(verify_key_contributions_request);

        let aggregate_request = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::VerifyAndAggregateKeyContributions(
                verify_req,
            )),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            2,
            aggregate_request.encode_to_vec().into(),
            Bytes::new(),
        );
        let response3 = advance_until_response(&mut cluster);
        let agg_pk_prost = match response3.msg {
            Some(reputable_decryptor_response::Msg::VerifyAndAggregateKeyContributions(resp)) => {
                resp.public_key.unwrap()
            }
            _ => panic!("Expected VerifyAndAggregateKeyContributionsResponse, got {:?}", response3),
        };

        // 3. Locally encrypt valid standard plaintext under key_1's public key
        let vahe =
            ShellVahe::new(create_shell_ahe_config(1).unwrap(), b"willow_reputable_decryptor")
                .unwrap();
        let agg_pk_pb: shell_ciphertexts_rust_proto::ShellAhePublicKey =
            convert_from_prost(&agg_pk_prost).unwrap();
        let agg_pk = <ShellVahe as AheBase>::PublicKey::from_proto(agg_pk_pb, &vahe).unwrap();

        let plaintext: Vec<i64> = vec![0; 8];
        let mut prng = <ShellVahe as AheBase>::Rng::create(
            &<ShellVahe as AheBase>::Rng::generate_seed().unwrap(),
        )
        .unwrap();
        let (ciphertext, _) =
            vahe.verifiable_encrypt(&plaintext, &agg_pk, b"test_nonce", &mut prng).unwrap();

        let pd_ciphertext_pb = vahe.get_partial_dec_ciphertext(&ciphertext).unwrap();
        let pd_ciphertext_pb_proto = pd_ciphertext_pb.to_proto(&vahe).unwrap();
        let pd_ciphertext_prost: ProstShellAhePartialDecCiphertext =
            convert_to_prost(&pd_ciphertext_pb_proto).unwrap();

        // 4. Perform successful partial decryption
        let mut dec_req = HandlePartialDecryptionRequest::default();
        dec_req.key_id = "key_1".to_string();
        let mut inner_dec_req = PartialDecryptionRequest::default();
        inner_dec_req.partial_dec_ciphertext = Some(pd_ciphertext_prost.clone());
        dec_req.partial_decryption_request = Some(inner_dec_req);

        let dec_request = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::HandlePartialDecryption(dec_req)),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            3,
            dec_request.encode_to_vec().into(),
            Bytes::new(),
        );
        let resp = advance_until_response(&mut cluster);
        match resp.msg {
            Some(reputable_decryptor_response::Msg::HandlePartialDecryption(resp)) => {
                let partial_dec = resp.partial_decryption_response.unwrap();
                assert!(partial_dec.partial_decryption.is_some());
            }
            _ => panic!("Expected HandlePartialDecryptionResponse, got {:?}", resp),
        }

        // 5. Send second decryption command for the same key, which must now fail.
        let mut dec_req_fail = HandlePartialDecryptionRequest::default();
        dec_req_fail.key_id = "key_1".to_string();
        let mut inner_dec_req_fail = PartialDecryptionRequest::default();
        inner_dec_req_fail.partial_dec_ciphertext = Some(pd_ciphertext_prost);
        dec_req_fail.partial_decryption_request = Some(inner_dec_req_fail);

        let dec_request_fail = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::HandlePartialDecryption(dec_req_fail)),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            4,
            dec_request_fail.encode_to_vec().into(),
            Bytes::new(),
        );

        let response_fail = advance_until_response(&mut cluster);
        match response_fail.msg {
            Some(reputable_decryptor_response::Msg::Error(status)) => {
                assert_eq!(status.code, 9); // FailedPrecondition
                assert!(status.message.contains("key_1 not found"));
            }
            _ => panic!("Expected Error response, got {:?}", response_fail),
        }
    }

    #[test]
    fn test_concurrent_key_setup_idempotency() {
        let mut cluster = FakeCluster::new(Bytes::new());
        cluster.start_node(
            1,
            true,
            ReputableDecryptorActor::new(
                DEFAULT_MAX_NUMBER_OF_DECRYPTORS,
                DEFAULT_MAX_NUMBER_OF_KEYS,
            ),
        );
        cluster.advance_until_elected_leader(None);

        let setup_req = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::CreateSetupContribution(
                CreateSetupContributionRequest { key_id: "concurrent_key".to_string() },
            )),
        };

        // Send two concurrent requests
        cluster.send_app_message(
            cluster.leader_id(),
            1,
            setup_req.encode_to_vec().into(),
            Bytes::new(),
        );
        cluster.send_app_message(
            cluster.leader_id(),
            2,
            setup_req.encode_to_vec().into(),
            Bytes::new(),
        );

        // Advance and get both responses
        let mut resp1: Option<ReputableDecryptorResponse> = None;
        let mut resp2: Option<ReputableDecryptorResponse> = None;

        cluster.advance_until(&mut |envelope_out| match &envelope_out.msg {
            Some(out_message::Msg::DeliverAppMessage(message)) => {
                let response =
                    ReputableDecryptorResponse::decode(message.message_header.as_ref()).unwrap();
                if message.correlation_id == 1 {
                    resp1 = Some(response);
                } else if message.correlation_id == 2 {
                    resp2 = Some(response);
                }
                return resp1.is_some() && resp2.is_some();
            }
            _ => false,
        });

        let c1 = match resp1.unwrap().msg {
            Some(reputable_decryptor_response::Msg::CreateSetupContribution(resp)) => {
                resp.setup_contribution.unwrap()
            }
            _ => panic!("Expected CreateSetupContributionResponse"),
        };

        let c2 = match resp2.unwrap().msg {
            Some(reputable_decryptor_response::Msg::CreateSetupContribution(resp)) => {
                resp.setup_contribution.unwrap()
            }
            _ => panic!("Expected CreateSetupContributionResponse"),
        };

        assert_eq!(
            c1.key_contribution.as_ref().unwrap().public_key_share.as_ref().unwrap().poly,
            c2.key_contribution.as_ref().unwrap().public_key_share.as_ref().unwrap().poly,
            "Concurrent setup contributions must return identical public key share"
        );
    }
}
