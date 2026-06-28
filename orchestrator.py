import json

# SYSTEM PROMPTS
PLANNER_PROMPT = "You are the PLANNER Agent. Break down the user query."
RESEARCHER_PROMPT = "You are the RESEARCHER Agent."
WRITER_PROMPT = "You are the WRITER Agent."
CRITIC_PROMPT = "You are the CRITIC Agent."

class VYOROrchestrator:
    def __init__(self, retrieval_fn=None):
        self.retrieval_fn = retrieval_fn if retrieval_fn else self._mock_qdrant_search
        self.max_debate_iterations = 3

    def _mock_qdrant_search(self, query: str) -> list[str]:
        return ["Source: HR_Policy_2026.pdf | Medical leaves are capped at 15 days per calendar year. Approval from Lead is mandatory."]

    def _mock_llm_call(self, system_prompt: str, user_content: str) -> str:
        if "PLANNER" in system_prompt:
            return json.dumps({"steps": ["Search for medical leave policy limits", "Check approval workflows"]})
        
        elif "RESEARCHER" in system_prompt:
            return json.dumps({"extracted_facts": "Medical leaves = 15 days max. Needs Lead approval.", "source": "HR_Policy_2026.pdf"})
        
        elif "WRITER" in system_prompt:
            # Agar 'REJECTED' waala text feedback mein hai, matlab ye Iteration 2 hai!
            if "REJECTED" in user_content:
                return json.dumps({
                    "answer": "According to the HR Policy 2026, employees are entitled to a maximum of 15 medical leaves per calendar year, requiring mandatory approval from the Lead.",
                    "citations": ["HR_Policy_2026.pdf"]
                })
            # Pehli baar mein vague answer dega jisme kuch nahi hoga
            return json.dumps({
                "answer": "You get some days off for medical reasons but you must ask your boss first.",
                "citations": []
            })
            
        elif "CRITIC" in system_prompt:
            # Agar draft mein sahi quantitative values hain, tabhi APPROVE hoga
            if "15 days" in user_content:
                return "APPROVED"
            else:
                return "REJECTED: Draft lacks precise metrics (15 days) and structural source citations."
        
        return "{}"

    def execute_query(self, user_query: str) -> dict:
        print(f"\n[Orchestrator]: Ingesting user query -> '{user_query}'")
        
        planner_res = self._mock_llm_call(PLANNER_PROMPT, user_query)
        plan = json.loads(planner_res).get("steps", [])
        print(f"[Planner] Sub-steps identified: {plan}")
        
        retrieved_context = self.retrieval_fn(user_query)
        print(f"[Qdrant Connector] Context Retrieved: {retrieved_context}")
        
        researcher_res = self._mock_llm_call(RESEARCHER_PROMPT, f"Context: {retrieved_context}")
        facts = json.loads(researcher_res)
        
        iteration = 0
        current_answer_dict = {}
        is_approved = False
        feedback = "Initial Draft Construction"
        
        while iteration < self.max_debate_iterations and not is_approved:
            iteration += 1
            print(f" [Debate Loop] Iteration {iteration}/{self.max_debate_iterations} active...")
            
            writer_input = f"Facts: {facts} | Feedback from Critic: {feedback}"
            writer_res = self._mock_llm_call(WRITER_PROMPT, writer_input)
            current_answer_dict = json.loads(writer_res)
            
            critic_res = self._mock_llm_call(CRITIC_PROMPT, f"Draft: {writer_res}")
            print(f" [Critic Evaluation]: {critic_res}")
            
            if critic_res.strip().upper() == "APPROVED":
                is_approved = True
                print(" [Critic Status]: Verification successful! Formatting final output packet.")
            else:
                feedback = critic_res
        
        confidence_score = 0.95 if is_approved else 0.40
        final_answer = current_answer_dict.get("answer", "")
        final_citations = current_answer_dict.get("citations", [])
        
        return {
            "answer": final_answer,
            "citations": final_citations,
            "confidence": confidence_score
        }