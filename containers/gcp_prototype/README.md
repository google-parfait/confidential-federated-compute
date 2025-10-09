# A prototype container that runs on GCP confidential space

This container is a reference container for exploration or testing.

How to build and upload:

```
bazelisk run :tarball
docker tag gcp_prototype:latest us-docker.pkg.dev/$PROJECT/$REPO/gcp_prototype:latest
docker push us-docker.pkg.dev/$PROJECT/$REPO/gcp_prototype:latest
```
