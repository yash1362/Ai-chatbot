# Ai-chatbot
Building an Ai agent
import asyncio
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from litellm import acompletion

app = FastAPI(title="Mixture of Agents Router")

# 1. Define the models you want to use as your "Proposers"
PROPOSER_MODELS = [
    "gpt-4o-mini",                  # Fast OpenAI model
    "claude-3-haiku-20240307",      # Fast Anthropic model
    "groq/llama3-8b-8192"           # Extremely fast open-source model on Groq
]

# 2. Define your heavyweight "Judge" model
JUDGE_MODEL = "gemini/gemini-1.5-pro-latest"

class ChatRequest(BaseModel):
    query: str

# Helper function to call a single LLM asynchronously
async def fetch_llm_response(model_name: str, user_query: str) -> dict:
    try:
        response = await acompletion(
            model=model_name,
            messages=[{"role": "user", "content": user_query}],
            temperature=0.7,
            timeout=10 # Strict timeout so one slow API doesn't hang the system
        )
        return {
            "model": model_name,
            "answer": response.choices[0].message.content
        }
    except Exception as e:
        # If a model fails or times out, return the error string as its "answer"
        return {
            "model": model_name, 
            "answer": f"FAILED: {str(e)}"
        }

@app.post("/chat")
async def process_chat(request: ChatRequest):
    user_query = request.query
    
    # ---------------------------------------------------------
    # STEP 1: The Fan-Out (Parallel Execution)
    # ---------------------------------------------------------
    proposer_results = []
    
    # TaskGroup cleanly manages concurrent tasks and handles exceptions gracefully
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(fetch_llm_response(model, user_query)) 
            for model in PROPOSER_MODELS
        ]
    
    # Extract the results once all tasks are complete
    proposer_results = [task.result() for task in tasks]

    # Format the collected answers into a string for the Judge
    formatted_proposals = ""
    for idx, result in enumerate(proposer_results):
        formatted_proposals += f"\n--- Chatbot {idx + 1} ({result['model']}) ---\n"
        formatted_proposals += f"{result['answer']}\n"

    # ---------------------------------------------------------
    # STEP 2: The LLM-as-a-Judge Evaluation
    # ---------------------------------------------------------
    
    # Inject the user query and the collected answers into the Judge template
    system_prompt = f"""
    You are an expert Aggregator and Evaluator model. Your task is to review multiple proposed answers to a single user query, score them, and synthesize the ultimate best response.

    # 1. The Input
    - User Query: {user_query}
    - Proposed Answers: 
    {formatted_proposals}

    # 2. Evaluation Criteria (Reasoning Step)
    For each chatbot's response, evaluate accuracy, completeness, and formatting.

    # 3. Scoring Rule
    - 0 (Fail): Wrong, irrelevant, or hallucinates facts.
    - 1 (Partial): Somewhat correct but misses key details.
    - 2 (Pass): Accurate, complete, and highly helpful.

    # 4. Constraints
    - Conflict Resolution: If the chatbots disagree, reason from first principles to determine the truth.
    - Output formatting: Do NOT mention that you are synthesizing multiple responses in the final answer.

    # 5. Output Format
    You must output a valid JSON object strictly matching this schema:
    {{
      "evaluations": [
        {{ "chatbot": "1", "model": "<model_name>", "reasoning": "<evaluation>", "score": <0,1,2> }}
      ],
      "best_chatbot": "<1, 2, or 3>",
      "synthesized_answer": "<The final, perfect answer>"
    }}
    """

    try:
        # Call the Judge model and force it to return JSON
        judge_response = await acompletion(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": system_prompt}],
            response_format={"type": "json_object"}, # Enforces JSON output
            temperature=0.2 # Low temperature for more analytical grading
        )
        
        raw_json_str = judge_response.choices[0].message.content
        final_output = json.loads(raw_json_str)
        
        return final_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Judge Evaluation Failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
