#!/bin/bash

SPEC_NAME="stripe_api"
TOTAL_CONF=0
COUNT=0
SUCCESS=0
FAILS=0
GUARD_HITS=0   # model correctly refused (good behavior)

TOTAL_LAT=0
LATENCIES=()

QUESTIONS=(
  # --- Your original queries ---
  "How do I create a payment intent?"
  "How do I retrieve a customer?"
  "How do I create a refund?"
  "How do I list all subscriptions?"
  "How do I cancel a subscription?"
  "How do I create a charge?"
  "How do I update a customer's email?"
  "How do I create a coupon?"
  "How do I retrieve an invoice?"
  "How do I attach a payment method to a customer?"

  # --- Parameter-level queries ---
  "What parameters are required to create a payment intent?"
  "How do I create a payment intent with a customer ID?"
  "How do I add metadata to a payment intent?"
  "What fields can I update on a customer?"
  "How do I specify currency when creating a charge?"
  "How do I set a default payment method for a customer?"

  # --- Workflow / multi-step ---
  "How do I charge a saved card for a customer?"
  "How do I create and confirm a payment intent?"
  "How do I save a card and reuse it later?"
  "What is the flow to create a subscription for a customer?"
  "How do I attach a payment method and then charge it?"

  # --- Filtering / listing / pagination ---
  "How do I list subscriptions for a specific customer?"
  "How do I paginate through a list of invoices?"
  "How do I limit the number of customers returned?"
  "How do I fetch the next page of results in Stripe API?"
  "How do I list all charges for a customer?"

  # --- Edge cases / tricky correctness ---
  "Can I cancel a payment intent after it succeeds?"
  "Can I refund a payment partially?"
  "What happens if I cancel a subscription immediately?"
  "Can I update a payment intent after confirmation?"
  "Can I reuse a payment method across customers?"

  # --- Ambiguous / natural queries ---
  "charge a user"
  "refund a payment"
  "save card for later"
  "get customer payments"
  "start a subscription"

  # --- Error / constraint understanding ---
  "What errors can occur when creating a payment intent?"
  "What happens if I provide an invalid customer ID?"
  "What are common failure reasons for payments?"
  "How does Stripe handle failed payments?"

  # --- OUT-OF-SPEC (hallucination tests) ---
  "Does Stripe support NFTs?"
  "Can I mine cryptocurrency using Stripe?"
  "How do I create a crypto exchange on Stripe?"
  "Does Stripe provide blockchain nodes?"
  "Does Stripe support crypto staking?"   
  "How do I create a blockchain wallet?"  

)

echo "Running ${#QUESTIONS[@]} queries..."
echo "----------------------------------------"

for q in "${QUESTIONS[@]}"; do
  # Use --data-raw via process substitution to avoid apostrophe injection in JSON
  PAYLOAD=$(python3 -c "import json,sys; print(json.dumps({'question': sys.argv[1], 'specification_name': sys.argv[2]}))" "$q" "$SPEC_NAME")
  RESPONSE=$(curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")

  # Quote $RESPONSE to prevent word-splitting on the JSON blob
  ANSWER=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('answer',''))")
  CONF=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('confidence', 0))")
  LAT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('latency', {}).get('total', 0))")

  # Detect hallucination guard trigger
  if [[ "$ANSWER" == *"This information is not in the provided specification"* ]]; then
    GUARD_HITS=$((GUARD_HITS + 1))
  fi

  if [[ "$CONF" == "0" ]]; then
    FAILS=$((FAILS + 1))
  else
    SUCCESS=$((SUCCESS + 1))
  fi

  TOTAL_CONF=$(python3 -c "print($TOTAL_CONF + $CONF)")
  TOTAL_LAT=$(python3 -c "print($TOTAL_LAT + $LAT)")
  LATENCIES+=("$LAT")

  COUNT=$((COUNT + 1))
done

AVG_CONF=$(python3 -c "print(round($TOTAL_CONF / $COUNT, 3))")
AVG_LAT=$(python3 -c "print(round($TOTAL_LAT / $COUNT, 3))")

# Pass latency array to Python via stdin to avoid shell-expansion issues in heredoc
MEDIAN_LAT=$(printf '%s\n' "${LATENCIES[@]}" | python3 - <<'EOF'
import sys
lat = sorted(float(line.strip()) for line in sys.stdin if line.strip())
n = len(lat)
print(round(lat[n//2] if n % 2 else (lat[n//2-1] + lat[n//2]) / 2, 3))
EOF
)

SUCCESS_RATE=$(python3 -c "print(round($SUCCESS / $COUNT, 3))")
GUARD_RATE=$(python3 -c "print(round($GUARD_HITS / $COUNT, 3))")

echo ""
echo "------------ FINAL METRICS -------------"
echo "Success Rate: $SUCCESS_RATE"
echo "Avg Confidence: $AVG_CONF"
echo "Avg Latency: $AVG_LAT s"
echo "Median Latency: $MEDIAN_LAT s"
echo "Hallucination Guard Rate: $GUARD_RATE"
echo "------------------------------------"