#!/usr/bin/env bash

set -euo pipefail

# Keep pre-seeded knowledge graph rules (entity_relations are cleared)
echo ">>> Preserving pre-seeded knowledge graph rules, clearing entity relations..."
python3 -c "
import json
with open('runtime_knowledge_graph.json') as f:
    kg = json.load(f)
kg['entity_relations'] = []
with open('runtime_knowledge_graph.json', 'w') as f:
    json.dump(kg, f, indent=2)
print(f'Kept {len(kg[\"rules\"])} pre-seeded rules, cleared entity relations.')
" 2>/dev/null || echo "Knowledge graph preserved as-is."
echo ""

PROJECT_ID="${GCP_PROJECT_ID:-ai-nm26osl-1735}"
REGION="${GCP_REGION:-europe-west4}"
SERVICE_NAME="${SERVICE_NAME:-tripletex-agent-yassy-auto}"
REPO_NAME="${REPO_NAME:-tripletex-agent-v3}"
RUNTIME_SERVICE_ACCOUNT="${RUNTIME_SERVICE_ACCOUNT:-tripletex-agent-sa}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-pro-preview}"
AGENT_API_KEY="${AGENT_API_KEY:-}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "GCP_PROJECT_ID is not set and no active gcloud project was found."
  exit 1
fi

SA_EMAIL="${RUNTIME_SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)"

echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo "Image: ${IMAGE}"
echo ""

echo ">>> Enabling required APIs"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  iam.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"

echo ">>> Ensuring runtime service account exists"
gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud iam service-accounts create "${RUNTIME_SERVICE_ACCOUNT}" \
  --display-name="Tripletex agent runtime" \
  --project="${PROJECT_ID}"

echo ">>> Ensuring Artifact Registry repository exists"
gcloud artifacts repositories describe "${REPO_NAME}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --project="${PROJECT_ID}"

echo ">>> Building container remotely with Cloud Build"
# The context is current directory, openapi.json is inside Docs/ already
gcloud builds submit \
  --tag "${IMAGE}" \
  --project="${PROJECT_ID}"

ENV_VARS="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GEMINI_MODEL=${GEMINI_MODEL}"
if [[ -n "${AGENT_API_KEY}" ]]; then
  ENV_VARS="${ENV_VARS},AGENT_API_KEY=${AGENT_API_KEY}"
fi

if [[ -n "${GEMINI_API_KEY:-}" ]]; then
  ENV_VARS="${ENV_VARS},GEMINI_API_KEY=${GEMINI_API_KEY}"
fi

# We don't need KNOWLEDGE_GCS_BUCKET directly yet as V3 defaults to local JSON persist, 
# but if needed, we'll append it later.

echo ">>> Deploying to Cloud Run"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --allow-unauthenticated \
  --service-account "${SA_EMAIL}" \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 1 \
  --timeout 540 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "${ENV_VARS}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format='value(status.url)')"

echo ""
echo "Deployment complete."
echo "Submit this endpoint:"
echo "  ${SERVICE_URL}/solve"
if [[ -n "${AGENT_API_KEY}" ]]; then
  echo ""
  echo "Use this API key in the competition platform:"
  echo "  ${AGENT_API_KEY}"
fi
echo ""
