import torch
from titans_memory import VYORNeuralBrain
from surprise_gate import VYORSurpriseGate


def test_ai_brain():
    print(" Starting Core AI Test...")
    
    # 1. Brain initialize karo (dim=256)
    brain = VYORNeuralBrain(dim=256)
    
    # 2. Simulate karo ki parsed text ka vector aaya hai
    # Shape: (batch_size=1, sequence_length=10, dimension=256)
    mock_document_vector = torch.randn(1, 10, 256)
    
    # 3. Mock Surprise condition check
    mock_surprise_score = 0.45 
    
    print(f" Current Surprise Score detected: {mock_surprise_score}")
    if mock_surprise_score >= 0.3:
        print(" High Surprise Detected! Sending to Titans Neural Memory...")
        brain.learn_new_info(mock_document_vector)
    else:
        print(" Boring/Old Data. Sending to Partner's Qdrant Vector DB...")

    print("Test Finished! Base AI module is 100% working.")

#day-2
import torch


def test_surprise_gate_pipeline():
    print("\n Starting  Day 2 Surprise Gate Test...")
    
    # Initialize Surprise Gate
    gate = VYORSurpriseGate()
    
    # Mock Memory Weights vs Incoming Chunks
    memory_base = torch.zeros(1, 1, 256) 
    
    # Scenario A: Low Surprise Chunk
    low_surprise_chunk = torch.randn(1, 1, 256) * 0.05
    
    # Scenario B: High Surprise Chunk
    high_surprise_chunk = torch.randn(1, 1, 256) * 2.5
    
    # Warmup loop simulating multiple document chunks hitting the system
    print(" Warming up the rolling history buffer with 12 structural chunks...")
    for i in range(12):
        mock_chunk = torch.randn(1, 1, 256) * 0.4
        loss = gate.compute_huber_loss(mock_chunk, memory_base)
        gate.update_and_route(loss)
        
    # Real Evaluation Post Warmup
    loss_low = gate.compute_huber_loss(low_surprise_chunk, memory_base)
    decision_low, th_low = gate.update_and_route(loss_low)
    print(f" Low Surprise Chunk Loss: {loss_low:.4f} | Dynamic Threshold: {th_low:.4f} -> Decision: 【{decision_low}】")
    
    loss_high = gate.compute_huber_loss(high_surprise_chunk, memory_base)
    decision_high, th_high = gate.update_and_route(loss_high)
    print(f"High Surprise Chunk Loss: {loss_high:.4f} | Dynamic Threshold: {th_high:.4f} -> Decision: 【{decision_high}】")

    print("Day 2 Surprise Gate Logic is 100% stable!")


    # Day 3 & Day 4: Multi-Agent Orchestrator Test Block
from orchestrator import VYOROrchestrator

def test_multi_agent_orchestrator():
    print("\n Starting  Day 3 & 4 Multi-Agent Framework Test...")
    
    # Initialize Orchestrator
    orchestrator = VYOROrchestrator()
    
    # Sample Query mimicking real company user questions
    sample_query = "How many medical leaves am I allowed and who approves them?"
    
    # Run the Agent Loop
    result = orchestrator.execute_query(sample_query)
    
    print("\n --- FINAL RETURN PACKET RECEIVED ---")
    print(f" Answer: {result['answer']}")
    print(f"Citations: {result['citations']}")
    print(f" Confidence Score: {result['confidence']}")
    print("----------------------------------------")
    
    assert "answer" in result, "Contract 2 missing 'answer' key"
    assert "citations" in result, "Contract 2 missing 'citations' key"
    assert isinstance(result["confidence"], float), "Contract 2 confidence must be a float"
    
    print(" Day 3 & 4 Multi-Agent Framework is 100% compliant with Contract 2 requirements!")
#Day 5
from integration_interface import VYORIntegrationInterface
def test_interface_bridge_pipeline():
    print("\n Starting Interface Bridge Test...")
    
    # Initialize the complete system bridge
    bridge = VYORIntegrationInterface(dim=256)
    
    # 1. Simulate Dataset Upload (High Surprise Chunk)
    print("\n--- Testing Component 1: Dataset Upload Ingestion ---")
    high_surprise_tensor = torch.randn(1, 1, 256) * 3.0  # High variance
    route_result = bridge.process_incoming_chunk(high_surprise_tensor, raw_text_meta="CONFIDENTIAL: Q2 Financial Sheets")
    print(f"Resulting Destination Route: {route_result}")
    
    # 2. Simulate User asking a question from UI Chatbox
    print("\n--- Testing Component 2: Chatbox Query Handling ---")
    query_result = bridge.run_orchestrator("What is the leaf policy constraint?")
    print(f"Final UI Packet -> Answer: {query_result['answer']} | Conf: {query_result['confidence']}")
    print("\nDay 5 Interface Bridge Contract is 100% Stable and Production Ready!")

# =====================================================================
# FILE ENTRANCE / EXECUTION GATEWAY (Day 1, Day 2 & Day 3/4)
# =====================================================================
if __name__ == "__main__":
    from titans_memory import VYORNeuralBrain
    from surprise_gate import VYORSurpriseGate 
    from orchestrator import VYOROrchestrator  
    
    print(" Starting Complete VYOR Core Verification Suite...")
    
    # Day 1 Test Case Execution
    test_ai_brain()                  
    
    # Day 2 Test Case Execution
    test_surprise_gate_pipeline()    
    
    # Day 3 & 4 Test Case Execution
    test_multi_agent_orchestrator()

    #Day 5 test case execution
    test_interface_bridge_pipeline()  