import torch
from titans_memory import VYORNeuralBrain
from surprise_gate import VYORSurpriseGate
from orchestrator import VYOROrchestrator

class VYORIntegrationInterface:
    def __init__(self, partner_qdrant_search_fn=None, partner_qdrant_insert_fn=None):
        """
        Day 5 Bridge Interface Initializer.
        partner_qdrant_search_fn: Kal Partner ka Qdrant Search function yahan pass hoga.
        partner_qdrant_insert_fn: Kal Partner ka Qdrant Data Insertion function yahan pass hoga.
        """
        print("[Bridge Interface]: Initializing Core VYOR AI System Connection...")
        
        # 1. Day 1 & Day 2 Components Initalize
        self.brain = VYORNeuralBrain(dim=256)
        self.gate = VYORSurpriseGate()
        
        # 2. External Database Connectors Bind (Kal use honge)
        self.partner_insert = partner_qdrant_insert_fn if partner_qdrant_insert_fn else self._fallback_insert
        self.partner_search = partner_qdrant_search_fn
        
        # 3. Day 3 & 4 Orchestrator Connect
        # Hum orchestrator ko partner ka search function pass kar dete hain
        self.orchestrator = VYOROrchestrator(retrieval_fn=self.partner_search)

    def _fallback_insert(self, chunk_data):
        """Mock behavior for Saturday testing when partner DB is absent"""
        print(" [Fallback DB]: Chunk saved to Static Partner Vector Store (Mock Qdrant).")
        return True

    # =====================================================================
    # FUNCTION 1: DATASET INGESTION PIPELINE (For File Uploads)
    # =====================================================================
    def process_incoming_chunk(self, chunk_tensor: torch.Tensor, raw_text_meta: str = "") -> str:
        """
        Partner ka code text chunks ka tensor bana kar isme pass karega.
        Surprise Gate loss check karega aur dynamic routing decision lega.
        """
        print(f"\n [Bridge]: Processing incoming chunk... Metadata: '{raw_text_meta[:30]}...'")
        
        # Create a blank memory base matrix for loss evaluation
        memory_base = torch.zeros_like(chunk_tensor)
        
        # 1. Compute Huber Loss via Surprise Gate
        loss = self.gate.compute_huber_loss(chunk_tensor, memory_base)
        
        # 2. Get Dynamic Threshold Routing Decision
        decision, threshold = self.gate.update_and_route(loss)
        print(f"[Gate Evaluation] Loss: {loss:.4f} | Dynamic Threshold: {threshold:.4f}")
        
        # 3. Execution Routing
        if decision == "memory_updated":
            print("[Route -> LTM]: High Surprise Detected! Updating Titans Neural Brain weights...")
            self.brain.learn_new_info(chunk_tensor)
            return "TITANS_NEURAL_MEMORY"
        else:
            print("[Route -> Qdrant]: Low Surprise Data. Offloading to Partner's Qdrant Vector Store...")
            self.partner_insert(raw_text_meta)
            return "PARTNER_QDRANT_DB"

    # =====================================================================
    # FUNCTION 2: USER QUERY ORCHESTRATION PIPELINE (For UI Chatbox)
    # =====================================================================
    def run_orchestrator(self, user_query: str) -> dict:
        """
        Streamlit UI jab user query bhejega, toh yeh function trigger hoga.
        Yeh multi-agent debate loop chalakar final clean dictionary return karega.
        """
        print(f"\n [Bridge]: User query received -> '{user_query}'")
        
        # Trigger Day 3 & 4 multi-agent flow
        final_packet = self.orchestrator.execute_query(user_query)
        
        print(" [Bridge]: Sending certified output packet back to UI layer.")
        return final_packet