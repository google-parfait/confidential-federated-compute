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
extern crate willow_committee_selector_service;
extern crate willow_multi_decryptor;
extern crate willow_reputable_decryptor_service;

mod test {
    use prost::bytes::Bytes;
    use prost::Message;
    use tcp_integration::harness::*;
    use tcp_proto::runtime::endpoint::out_message;
    use willow_committee_selector_service::apps::willow::committee_selector::service::{
        committee_selector_request, committee_selector_response, CommitteeSelectorRequest,
        CreateCommitteeRequest,
    };
    use willow_multi_decryptor::multi_decryptor::service::{
        multi_decryptor_request, multi_decryptor_response, MultiDecryptorConfig,
        MultiDecryptorRequest, MultiDecryptorResponse,
    };
    use willow_multi_decryptor::MultiDecryptorActor;
    use willow_reputable_decryptor_service::apps::willow::reputable_decryptor::service::{
        reputable_decryptor_request, reputable_decryptor_response, CreateSetupContributionRequest,
        ReputableDecryptorRequest,
    };

    fn advance_until_response(
        cluster: &mut FakeCluster<MultiDecryptorActor>,
    ) -> MultiDecryptorResponse {
        let mut multi_decryptor_response: Option<MultiDecryptorResponse> = None;
        let _response_messages =
            cluster.advance_until(&mut |envelope_out| match &envelope_out.msg {
                Some(out_message::Msg::DeliverAppMessage(message)) => {
                    let response =
                        MultiDecryptorResponse::decode(message.message_header.as_ref()).unwrap();
                    multi_decryptor_response = Some(response);
                    return true;
                }
                _ => false,
            });

        assert!(multi_decryptor_response.is_some());
        multi_decryptor_response.unwrap()
    }

    #[test]
    fn test_multi_decryptor_actor_routing() {
        let config = MultiDecryptorConfig {
            max_number_of_committees: 128,
            max_number_of_decryptors: 128,
            max_number_of_keys: 100,
        };
        let mut config_bytes = Vec::new();
        config.encode(&mut config_bytes).unwrap();
        let mut cluster = FakeCluster::new(config_bytes.into());

        // Construct MultiDecryptorActor with insecure reference values and default
        // limits
        let actor = MultiDecryptorActor::new_insecure();
        cluster.start_node(1, true, actor);
        cluster.advance_until_elected_leader(None);
        assert_eq!(cluster.leader_id(), 1);

        // 1. Test Committee Selector Request
        let cs_request = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CreateCommittee(CreateCommitteeRequest {
                committee_id: 123,
            })),
        };
        let multi_decryptor_request_cs = MultiDecryptorRequest {
            msg: Some(multi_decryptor_request::Msg::CommitteeSelector(cs_request)),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            1,
            multi_decryptor_request_cs.encode_to_vec().into(),
            Bytes::new(),
        );

        let response = advance_until_response(&mut cluster);
        match response.msg {
            Some(multi_decryptor_response::Msg::CommitteeSelector(cs_resp)) => match cs_resp.msg {
                Some(committee_selector_response::Msg::CreateCommittee(_)) => {}
                _ => panic!("Expected CreateCommitteeResponse, got {:?}", cs_resp),
            },
            _ => panic!("Expected CommitteeSelector response, got {:?}", response),
        }

        // 2. Test Reputable Decryptor Request (Error path to avoid unimplemented
        //    crypto)
        let rd_request = ReputableDecryptorRequest {
            msg: Some(reputable_decryptor_request::Msg::CreateSetupContribution(
                CreateSetupContributionRequest {
                    key_id: "".to_string(), // Empty key_id triggers InvalidArgument
                },
            )),
        };
        let multi_decryptor_request_rd = MultiDecryptorRequest {
            msg: Some(multi_decryptor_request::Msg::ReputableDecryptor(rd_request)),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            2,
            multi_decryptor_request_rd.encode_to_vec().into(),
            Bytes::new(),
        );

        let response = advance_until_response(&mut cluster);
        match response.msg {
            Some(multi_decryptor_response::Msg::ReputableDecryptor(rd_resp)) => match rd_resp.msg {
                Some(reputable_decryptor_response::Msg::Error(status)) => {
                    assert_eq!(status.code, 3); // InvalidArgument
                    assert!(status.message.contains("missing key_id field"));
                }
                _ => panic!("Expected Error response, got {:?}", rd_resp),
            },
            _ => panic!("Expected ReputableDecryptor response, got {:?}", response),
        }
    }
}
