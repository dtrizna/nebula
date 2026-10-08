import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from nebula import Nebula

def test_inference_determinism():
    print("Running inference determinism tests...")
    
    # 1. Test eval mode upon initialization
    nebula = Nebula(vocab_size=50000, seq_len=512, tokenizer="bpe")
    assert not nebula.model.training, "Nebula().model should be in eval() mode upon initialization!"
    print("  [PASS] Nebula() initializes in eval() mode.")

    # 2. Test repeated single-report inference consistency
    example_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "emulation",
        "report_example_entrypoint_with_registry.json"
    )
    with open(example_path, "r") as f:
        report = json.load(f)

    x = nebula.preprocess(report)
    probs = [nebula.predict_proba(x) for _ in range(5)]
    assert np.allclose(probs, probs[0]), f"Repeated predict_proba calls produced varying results: {probs}"
    print(f"  [PASS] Single-sample inference is deterministic: {probs[0]:.4f}")

    # 3. Test batch-slot invariance
    model = nebula.model
    xt = torch.as_tensor(x, dtype=torch.long)
    
    with torch.no_grad():
        score_alone = model(xt).squeeze().item()
        # Batch of 32 identical copies
        score_batch32 = model(xt.repeat(32, 1)).squeeze(-1).cpu().numpy()
        # Batch of 96 identical copies
        score_batch96 = model(xt.repeat(96, 1)).squeeze(-1).cpu().numpy()

    diff32 = np.abs(score_batch32 - score_alone).max()
    diff96 = np.abs(score_batch96 - score_alone).max()

    assert diff32 < 1e-5, f"Batch-32 max diff from singleton is {diff32}"
    assert diff96 < 1e-5, f"Batch-96 max diff from singleton is {diff96}"
    print(f"  [PASS] Batched inference matches singleton inference across all batch slots (max diff: {max(diff32, diff96):.2e}).")
    print("All determinism tests passed successfully!")

if __name__ == "__main__":
    test_inference_determinism()
