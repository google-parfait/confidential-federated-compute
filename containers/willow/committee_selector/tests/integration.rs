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

mod test {
    use prost::bytes::Bytes;
    use prost::Message;
    use tcp_integration::harness::*;
    use tcp_proto::runtime::endpoint::out_message;
    use willow_committee_selector_service::actor::CommitteeSelectorActor;
    use willow_committee_selector_service::apps::willow::committee_selector::service::{
        committee_selector_request, committee_selector_response, CheckCommitteeStatusRequest,
        CommitteeSelectorConfig, CommitteeSelectorRequest, CommitteeSelectorResponse,
        CommitteeStatus, CreateCommitteeRequest, SampleCommitteeRequest,
        VolunteerBatchForCommitteeRequest, VolunteerForCommitteeRequest,
    };

    const DEFAULT_MAX_NUMBER_OF_COMMITTEES: usize = 128;

    fn advance_until_response(
        cluster: &mut FakeCluster<CommitteeSelectorActor>,
    ) -> CommitteeSelectorResponse {
        let mut committee_selector_response: Option<CommitteeSelectorResponse> = None;
        let response_messages =
            cluster.advance_until(&mut |envelope_out| match &envelope_out.msg {
                Some(out_message::Msg::DeliverAppMessage(message)) => {
                    let response =
                        CommitteeSelectorResponse::decode(message.message_header.as_ref()).unwrap();
                    committee_selector_response = Some(response);
                    return true;
                }
                _ => false,
            });

        assert!(!response_messages.is_empty());
        committee_selector_response.unwrap()
    }

    fn new_test_cluster(limit: usize) -> FakeCluster<CommitteeSelectorActor> {
        let config = CommitteeSelectorConfig { max_number_of_committees: limit as i32 };
        let mut config_bytes = Vec::new();
        config.encode(&mut config_bytes).unwrap();
        FakeCluster::new(config_bytes.into())
    }

    #[test]
    fn test_create_committee() {
        let mut cluster = new_test_cluster(DEFAULT_MAX_NUMBER_OF_COMMITTEES);

        cluster.start_node(1, true, CommitteeSelectorActor::new());
        cluster.advance_until_elected_leader(None);
        assert!(cluster.leader_id() == 1);

        let create_request = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CreateCommittee(CreateCommitteeRequest {
                committee_id: 123,
            })),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            1,
            create_request.encode_to_vec().into(),
            Bytes::new(),
        );

        let response = advance_until_response(&mut cluster);
        match response.msg {
            Some(committee_selector_response::Msg::CreateCommittee(_)) => {}
            _ => panic!("Expected CreateCommitteeResponse, got {:?}", response),
        }
    }

    #[test]
    fn test_create_committee_idempotency() {
        let mut cluster = new_test_cluster(DEFAULT_MAX_NUMBER_OF_COMMITTEES);
        cluster.start_node(1, true, CommitteeSelectorActor::new());
        cluster.advance_until_elected_leader(None);

        let req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CreateCommittee(CreateCommitteeRequest {
                committee_id: 1,
            })),
        };

        cluster.send_app_message(cluster.leader_id(), 1, req.encode_to_vec().into(), Bytes::new());
        advance_until_response(&mut cluster);

        // Second call with same committee_id, using a different message ID (2)
        cluster.send_app_message(cluster.leader_id(), 2, req.encode_to_vec().into(), Bytes::new());
        let response = advance_until_response(&mut cluster);
        match response.msg {
            Some(committee_selector_response::Msg::CreateCommittee(_)) => {}
            _ => panic!("Expected CreateCommitteeResponse on duplicate call, got {:?}", response),
        }
    }

    #[test]
    fn test_create_committee_replaces_oldest() {
        let mut cluster = new_test_cluster(32);
        let actor = CommitteeSelectorActor::new();
        cluster.start_node(1, true, actor);
        cluster.advance_until_elected_leader(None);

        // MAX_NUMBER_OF_COMMITTEES is 32. Create 33.
        for i in 0..33 {
            let req = CommitteeSelectorRequest {
                msg: Some(committee_selector_request::Msg::CreateCommittee(
                    CreateCommitteeRequest { committee_id: i },
                )),
            };
            cluster.send_app_message(
                cluster.leader_id(),
                (i + 1) as u64,
                req.encode_to_vec().into(),
                Bytes::new(),
            );
            advance_until_response(&mut cluster);
        }

        let req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CheckCommitteeStatus(
                CheckCommitteeStatusRequest { committee_id: 0 },
            )),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            100,
            req.encode_to_vec().into(),
            Bytes::new(),
        );
        let response = advance_until_response(&mut cluster);
        match response.msg {
            Some(committee_selector_response::Msg::Error(err)) => {
                assert_eq!(err.code, 9);
                assert_eq!(err.message, "Committee not found for given 0 committee id");
            }
            _ => panic!("Expected Error since oldest committee should be removed"),
        }
    }

    #[test]
    fn test_volunteer_batch_for_committee() {
        let mut cluster = new_test_cluster(DEFAULT_MAX_NUMBER_OF_COMMITTEES);
        cluster.start_node(1, true, CommitteeSelectorActor::new());
        cluster.advance_until_elected_leader(None);

        let req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CreateCommittee(CreateCommitteeRequest {
                committee_id: 1,
            })),
        };
        cluster.send_app_message(cluster.leader_id(), 1, req.encode_to_vec().into(), Bytes::new());
        advance_until_response(&mut cluster);

        let vol_req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::VolunteerBatchForCommittee(
                VolunteerBatchForCommitteeRequest {
                    volunteers: vec![VolunteerForCommitteeRequest {
                        public_key: vec![1, 2, 3],
                        key_endorsement: vec![],
                        // To-do: add a valid 509 certificate in key_endorsement.
                    }],
                },
            )),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            2,
            vol_req.encode_to_vec().into(),
            Bytes::new(),
        );
        let response = advance_until_response(&mut cluster);

        match response.msg {
            Some(committee_selector_response::Msg::VolunteerBatchForCommittee(resp)) => {
                assert_eq!(resp.assignments.len(), 1);
                assert_eq!(resp.assignments[0].committee_id, 1);
            }
            _ => panic!("Expected VolunteerBatchForCommitteeResponse, got {:?}", response),
        }
    }

    #[test]
    fn test_volunteer_without_committee() {
        let mut cluster = new_test_cluster(DEFAULT_MAX_NUMBER_OF_COMMITTEES);
        cluster.start_node(1, true, CommitteeSelectorActor::new());
        cluster.advance_until_elected_leader(None);

        let vol_req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::VolunteerBatchForCommittee(
                VolunteerBatchForCommitteeRequest {
                    volunteers: vec![VolunteerForCommitteeRequest {
                        public_key: vec![1, 2, 3],
                        key_endorsement: vec![],
                    }],
                },
            )),
        };

        cluster.send_app_message(
            cluster.leader_id(),
            2,
            vol_req.encode_to_vec().into(),
            Bytes::new(),
        );
        let response = advance_until_response(&mut cluster);
        match response.msg {
            Some(committee_selector_response::Msg::VolunteerBatchForCommittee(resp)) => {
                assert_eq!(resp.assignments.len(), 1);
                assert_eq!(resp.assignments[0].committee_id, -1);
            }
            _ => panic!("Expected VolunteerBatchForCommitteeResponse, got {:?}", response),
        }
    }

    #[test]
    fn test_sample_committee_unknown() {
        let mut cluster = new_test_cluster(DEFAULT_MAX_NUMBER_OF_COMMITTEES);
        cluster.start_node(1, true, CommitteeSelectorActor::new());
        cluster.advance_until_elected_leader(None);

        let req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::SampleCommittee(SampleCommitteeRequest {
                committee_id: 1,
            })),
        };
        cluster.send_app_message(cluster.leader_id(), 1, req.encode_to_vec().into(), Bytes::new());
        let response = advance_until_response(&mut cluster);
        match response.msg {
            Some(committee_selector_response::Msg::Error(err)) => {
                assert_eq!(err.code, 9);
                assert_eq!(err.message, "Committee not found for committee id 1");
            }
            _ => panic!("Expected Error, got {:?}", response),
        }
    }

    #[test]
    fn test_sample_committee_not_enough_volunteers() {
        let mut cluster = new_test_cluster(DEFAULT_MAX_NUMBER_OF_COMMITTEES);
        cluster.start_node(1, true, CommitteeSelectorActor::new());
        cluster.advance_until_elected_leader(None);

        let req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CreateCommittee(CreateCommitteeRequest {
                committee_id: 2,
            })),
        };
        cluster.send_app_message(cluster.leader_id(), 1, req.encode_to_vec().into(), Bytes::new());
        advance_until_response(&mut cluster);

        let vol_req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::VolunteerBatchForCommittee(
                VolunteerBatchForCommitteeRequest {
                    volunteers: vec![VolunteerForCommitteeRequest {
                        public_key: vec![1],
                        key_endorsement: vec![],
                    }],
                },
            )),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            2,
            vol_req.encode_to_vec().into(),
            Bytes::new(),
        );
        advance_until_response(&mut cluster);

        let sample_req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::SampleCommittee(SampleCommitteeRequest {
                committee_id: 2,
            })),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            3,
            sample_req.encode_to_vec().into(),
            Bytes::new(),
        );
        let response = advance_until_response(&mut cluster);
        match response.msg {
            Some(committee_selector_response::Msg::Error(err)) => {
                assert_eq!(err.code, 9);
                assert_eq!(err.message, "Committee 2 does not have enough volunteers");
            }
            _ => panic!("Expected Error, got {:?}", response),
        }
    }

    #[test]
    fn test_sample_committee_success() {
        let mut cluster = new_test_cluster(DEFAULT_MAX_NUMBER_OF_COMMITTEES);
        cluster.start_node(1, true, CommitteeSelectorActor::new());
        cluster.advance_until_elected_leader(None);

        let req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CreateCommittee(CreateCommitteeRequest {
                committee_id: 1,
            })),
        };
        cluster.send_app_message(cluster.leader_id(), 1, req.encode_to_vec().into(), Bytes::new());
        advance_until_response(&mut cluster);

        // MIN_NUMBER_OF_VOLUNTEERS_PER_COMMITTEE is 1024 * 1024
        // Send 1025 batches of 1024 volunteers each.
        for i in 0..1025 {
            let mut volunteers = Vec::with_capacity(1024);
            for j in 0..1024 {
                volunteers.push(VolunteerForCommitteeRequest {
                    public_key: ((i * 1024 + j) as u32).to_le_bytes().to_vec(),
                    key_endorsement: vec![],
                });
            }

            let vol_req = CommitteeSelectorRequest {
                msg: Some(committee_selector_request::Msg::VolunteerBatchForCommittee(
                    VolunteerBatchForCommitteeRequest { volunteers },
                )),
            };

            cluster.send_app_message(
                cluster.leader_id(),
                (i + 2) as u64,
                vol_req.encode_to_vec().into(),
                Bytes::new(),
            );
            advance_until_response(&mut cluster);
        }

        let sample_req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::SampleCommittee(SampleCommitteeRequest {
                committee_id: 1,
            })),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            1027,
            sample_req.encode_to_vec().into(),
            Bytes::new(),
        );
        let response = advance_until_response(&mut cluster);
        match response.msg {
            Some(committee_selector_response::Msg::CheckCommitteeStatus(resp)) => {
                assert_eq!(resp.committee_id, 1);
                assert_eq!(resp.status, CommitteeStatus::SelectionComplete as i32);
                assert_eq!(resp.members.len(), 128);
            }
            _ => panic!("Expected CheckCommitteeStatus, got {:?}", response),
        }

        // Check sample again to test "Committee is not active" error
        cluster.send_app_message(
            cluster.leader_id(),
            1028,
            sample_req.encode_to_vec().into(),
            Bytes::new(),
        );
        let err_response = advance_until_response(&mut cluster);
        match err_response.msg {
            Some(committee_selector_response::Msg::Error(err)) => {
                assert_eq!(err.message, "Committee 1 is not active");
            }
            _ => panic!("Expected Error, got {:?}", err_response),
        }
    }

    #[test]
    fn test_check_committee_status_success() {
        let mut cluster = new_test_cluster(DEFAULT_MAX_NUMBER_OF_COMMITTEES);
        cluster.start_node(1, true, CommitteeSelectorActor::new());
        cluster.advance_until_elected_leader(None);

        let req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CreateCommittee(CreateCommitteeRequest {
                committee_id: 1,
            })),
        };
        cluster.send_app_message(cluster.leader_id(), 1, req.encode_to_vec().into(), Bytes::new());
        advance_until_response(&mut cluster);

        let check_req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CheckCommitteeStatus(
                CheckCommitteeStatusRequest { committee_id: 1 },
            )),
        };
        cluster.send_app_message(
            cluster.leader_id(),
            2,
            check_req.encode_to_vec().into(),
            Bytes::new(),
        );
        let response = advance_until_response(&mut cluster);
        match response.msg {
            Some(committee_selector_response::Msg::CheckCommitteeStatus(resp)) => {
                assert_eq!(resp.committee_id, 1);
                assert_eq!(resp.status, CommitteeStatus::AcceptingVolunteers as i32);
            }
            _ => panic!("Expected CheckCommitteeStatus, got {:?}", response),
        }
    }

    #[test]
    fn test_unknown_request() {
        let mut cluster = new_test_cluster(DEFAULT_MAX_NUMBER_OF_COMMITTEES);
        cluster.start_node(1, true, CommitteeSelectorActor::new());
        cluster.advance_until_elected_leader(None);

        let req = CommitteeSelectorRequest { msg: None };
        cluster.send_app_message(cluster.leader_id(), 1, req.encode_to_vec().into(), Bytes::new());
        let response = advance_until_response(&mut cluster);

        match response.msg {
            Some(committee_selector_response::Msg::Error(err)) => {
                assert_eq!(err.code, 3); // InvalidArgument
            }
            _ => panic!("Expected Error, got {:?}", response),
        }
    }

    #[test]
    fn test_volunteer_batch_randomness() {
        // Prepare 200 volunteers
        let mut volunteers = Vec::with_capacity(200);
        for i in 0..200 {
            volunteers.push(VolunteerForCommitteeRequest {
                public_key: vec![i as u8],
                key_endorsement: vec![],
            });
        }

        let vol_req = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::VolunteerBatchForCommittee(
                VolunteerBatchForCommitteeRequest { volunteers: volunteers.clone() },
            )),
        };

        // Run 1
        let mut cluster1 = new_test_cluster(DEFAULT_MAX_NUMBER_OF_COMMITTEES);
        cluster1.start_node(1, true, CommitteeSelectorActor::new());
        cluster1.advance_until_elected_leader(None);

        let req1 = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CreateCommittee(CreateCommitteeRequest {
                committee_id: 1,
            })),
        };
        cluster1.send_app_message(
            cluster1.leader_id(),
            1,
            req1.encode_to_vec().into(),
            Bytes::new(),
        );
        advance_until_response(&mut cluster1);

        cluster1.send_app_message(
            cluster1.leader_id(),
            2,
            vol_req.encode_to_vec().into(),
            Bytes::new(),
        );
        advance_until_response(&mut cluster1);

        let check_req1 = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CheckCommitteeStatus(
                CheckCommitteeStatusRequest { committee_id: 1 },
            )),
        };
        cluster1.send_app_message(
            cluster1.leader_id(),
            3,
            check_req1.encode_to_vec().into(),
            Bytes::new(),
        );
        let response1 = advance_until_response(&mut cluster1);

        let members1 = match response1.msg {
            Some(committee_selector_response::Msg::CheckCommitteeStatus(resp)) => resp.members,
            _ => panic!("Expected CheckCommitteeStatusResponse"),
        };

        // Run 2
        let mut cluster2 = new_test_cluster(DEFAULT_MAX_NUMBER_OF_COMMITTEES);
        cluster2.start_node(1, true, CommitteeSelectorActor::new());
        cluster2.advance_until_elected_leader(None);

        let req2 = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CreateCommittee(CreateCommitteeRequest {
                committee_id: 1, // Use same committee ID
            })),
        };
        cluster2.send_app_message(
            cluster2.leader_id(),
            1,
            req2.encode_to_vec().into(),
            Bytes::new(),
        );
        advance_until_response(&mut cluster2);

        cluster2.send_app_message(
            cluster2.leader_id(),
            3,
            vol_req.encode_to_vec().into(),
            Bytes::new(),
        );
        advance_until_response(&mut cluster2);

        let check_req2 = CommitteeSelectorRequest {
            msg: Some(committee_selector_request::Msg::CheckCommitteeStatus(
                CheckCommitteeStatusRequest { committee_id: 1 },
            )),
        };
        cluster2.send_app_message(
            cluster2.leader_id(),
            4,
            check_req2.encode_to_vec().into(),
            Bytes::new(),
        );
        let response2 = advance_until_response(&mut cluster2);

        let members2 = match response2.msg {
            Some(committee_selector_response::Msg::CheckCommitteeStatus(resp)) => resp.members,
            _ => panic!("Expected CheckCommitteeStatusResponse"),
        };

        // They should not be identical with high probability
        assert_ne!(members1, members2, "Committee members should be different due to randomness");
    }
}
