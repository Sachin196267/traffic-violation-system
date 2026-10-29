# signal_bridge - Python decides, Unity simulates

    python3 signal_bridge/server.py --controller pravah    # PRAVAH (exact port of the C# version)
    python3 signal_bridge/server.py --controller project   # modules/smart_signal_controller.py

Then in Unity pick **Python bridge** in the panel (or headless: `-executeMethod SignalSim.Experiment.Run -only 4`).
If the server is not running the junction falls back to a fixed 30 s cycle and the panel says "OFFLINE".
Protocol and message fields: top of `server.py`. Every decision is logged to `signal_bridge/logs/*.jsonl`.
Results and the port verification: `signal_sim/Results/bridge.md`.
