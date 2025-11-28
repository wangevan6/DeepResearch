from openai import OpenAI
import os

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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"
)

# First API call with reasoning
response = client.chat.completions.create(
  model="alibaba/tongyi-deepresearch-30b-a3b",
  messages=[
          {
            "role": "user",
            "content": "How can we unify JEPA-style predictive representations with tool-use agents (SPORT / FlowRL / RLVR) to create a single agent that can perform long-horizon planning using latent predictive rollouts?"
          }
        ],
  max_tokens=99482,  # Set maximum output tokens to 20,000
  extra_body={"reasoning": {"enabled": True}}
)

# Extract the assistant message with reasoning_details
response = response.choices[0].message

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "How can we unify JEPA-style predictive representations with tool-use agents (SPORT / FlowRL / RLVR) to create a single agent that can perform long-horizon planning using latent predictive rollouts??"},
  {
    "role": "assistant",
    "content": response.content,
    "reasoning_details": response.reasoning_details  # Pass back unmodified
  },
  {"role": "user", "content": "This does not make sense, i need more create training plan , and also what is the energy fucntion in your proposed plan"}
]

# Second API call - model continues reasoning from where it left off
response2 = client.chat.completions.create(
  model="alibaba/tongyi-deepresearch-30b-a3b",
  messages=messages,
  max_tokens=99482,  # Set maximum output tokens to 20,000
  extra_body={"reasoning": {"enabled": True}}
)

# Print results
print("First response:")
print(f"Content: {response.content}")
if hasattr(response, 'reasoning_details') and response.reasoning_details:
    print(f"Reasoning details: {response.reasoning_details}")

print("\nSecond response:")
print(f"Content: {response2.choices[0].message.content}")
if hasattr(response2.choices[0].message, 'reasoning_details') and response2.choices[0].message.reasoning_details:
    print(f"Reasoning details: {response2.choices[0].message.reasoning_details}")
