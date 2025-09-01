
#!/usr/bin/env python3
import os
import sys
import runpy

if __name__ == "__main__" and os.environ.get("STREAMLIT_RUN") != "1":
    # First invocation: re-launch under `streamlit run`
    os.environ["STREAMLIT_RUN"] = "1"
    sys.argv = ["streamlit", "run", __file__] + sys.argv[1:]
    from streamlit.web.cli import main as stcli
    sys.exit(stcli())

# From here on, we’re in Streamlit’s runtime.
# Instead of `import app`, exec the script fresh every rerun:
runpy.run_path(
    os.path.join(os.path.dirname(__file__), "app.py"),
    run_name="__main__"
)
