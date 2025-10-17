# A prototype container that runs on GCP confidential space

This container is a reference container for exploration or testing,
developed incremnetally, still in prototyping stages.

At the moment, the prototype is a generic "hello" server that listens
on Oak's Noise protocol-based gRPC endpoint, and replies hello, with
a matching simple test client that sends a hello message to verify that
the server works.

This is still evolving skeleton test setup, with the following
functionality:

1. Client sends hello, and server responds with a
   simple hello for now, just as in HTTP.

2. The communication is channeled through the
   Oak impementaiton of Noise, with proper
   handshake and encryption, albeit for now with
   no attestation, no verification, no binding,
   etc. configured (this will come in follow ups
   shortly).

3. Simple message pumps move messages between the
   state machienes and the gRPC stream.

4. Multiple client calls are properly handled.

5. For generality, we do not assume one-to-one
   correspondence between unencrypted requests
   from the client, and the replies from the
   server.

6. Client engages in a single exchnage, and then
   quits.

7. Server can handle multiple clients.

8. All incoming and outgoing messages are printed to visualize the
   dynamics protocol. We'll retain this debugging functionality
   until the implementation is complete.

Attestation and attestation verification will be added in the
follow ups.

How to build and run the server:

```
bazelisk run :tarball
docker tag gcp_prototype:latest us-docker.pkg.dev/$PROJECT/$REPO/gcp_prototype:latest
docker push us-docker.pkg.dev/$PROJECT/$REPO/gcp_prototype:latest
# launch the confidential VM from the image
```

How to build and run the client:

```
bazelisk run :test_client -- --server_address=<confidential VM's external address>
```
