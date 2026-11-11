"""Start the drawing coordinator on the deployed app, independent of this client.
    ../pilot/.venv/bin/python spawn_draw.py 0,1,2,3"""
import sys
import modal
f = modal.Function.from_name("dublin-metro-draw", "orchestrate")
call = f.spawn(sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3", "", "main", False)
print("spawned", call.object_id)
