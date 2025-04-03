import sys
from pathlib import Path
import importlib

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_import(import_path):
    try:
        module, obj = import_path.rsplit('.', 1)
        mod = importlib.import_module(module)
        getattr(mod, obj)
        print(f"✅ {import_path}")
        return True
    except Exception as e:
        print(f"❌ {import_path} - {type(e).__name__}: {str(e)}")
        return False

def check_file(path):
    full_path = PROJECT_ROOT / path
    if full_path.exists():
        print(f"✅ {path} exists")
        return True
    else:
        print(f"❌ {path} missing")
        return False

if __name__ == "__main__":
    print("=== Import Verification ===")
    imports_to_test = [
        "src.utils.logging.setup_logger",
        "src.models.transformer.trainer.TransformerTrainer",
        "src.models.rl.env.TradingEnv",  # Kept since env.py is in models/rl/
        "src.agents.rl_agent.RLAgent",   # Updated to agents folder
        "prometheus_client.Gauge",
        "stable_baselines3.SAC"
    ]

    print("\n=== File Existence Check ===")
    files_to_check = [
        "src/models/transformer/trainer.py",
        "src/agents/rl_agent.py",         # New location
        "src/models/rl/env.py",
        "src/utils/logging.py",
        "src/utils/normalization.py"
    ]

    print("\nTesting imports...")
    import_results = [test_import(i) for i in imports_to_test]

    print("\nChecking files...")
    file_results = [check_file(f) for f in files_to_check]

    print("\n=== Summary ===")
    print(f"Imports working: {sum(import_results)}/{len(import_results)}")
    print(f"Files found: {sum(file_results)}/{len(file_results)}")

    if all(import_results + file_results):
        print("🎉 All checks passed!")
    else:
        print("⚠️ Some issues detected - check above output")
        sys.exit(1)
