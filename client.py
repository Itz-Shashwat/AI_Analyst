import streamlit as st
import requests
import pandas as pd
import traceback
import re
import types

API_URL =  "https://startingly-mazier-anneliese.ngrok-free.dev/predict"

st.title("LLM-Powered Data Analysis")
st.write("-Shashwat Sharma")
st.write("Upload a CSV and enter a task. Note: Your data won't be uploaded to any server!.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
user_prompt = st.text_area("Enter your question",
                           placeholder="e.g., Get the average price of each product")

def extract_code(model_output: str) -> str:
    if not model_output:
        return ""
    fenced = re.search(r"```(?:python)?\n?(.*?)```", model_output, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    labeled = re.match(r"^\s*(?:response|resp|output|result|code)\s*(?:[:=]\s*)(.*)$", model_output, flags=re.IGNORECASE | re.DOTALL)
    if labeled:
        return labeled.group(1).strip()
    labeled_line = re.search(r"^[ \t-]*(?:response|resp|output|result|code)\s*(?:[:=]\s*)(.*)$", model_output, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if labeled_line:
        return labeled_line.group(1).strip()

    return model_output.strip()

def safe_execute(code_str: str, df: pd.DataFrame):
    local_vars = {"df": df, "__name__": "__main__"}
    try:
        compiled = compile(code_str, "<string>", "eval")
        out = eval(compiled, {}, local_vars)
        return True, out
    except SyntaxError:
        try:
            compiled = compile(code_str, "<string>", "exec")
            exec(compiled, {}, local_vars)
            if "result" in local_vars:
                return True, local_vars["result"]
            for candidate in ("out", "output", "df_out", "df_result"):
                if candidate in local_vars:
                    return True, local_vars[candidate]
            cleaned = {k: v for k, v in local_vars.items() if k not in ("__name__",)}
            return True, cleaned
        except Exception as e_exec:
            return False, traceback.format_exc()
    except Exception as e_eval:
        return False, traceback.format_exc()

if uploaded_file and user_prompt:

    df = pd.read_csv(uploaded_file)
    st.subheader("Data Columns")
    st.dataframe(df.columns)
    col_list = ", ".join(df.columns)
    final_prompt = f"Question: {user_prompt}\nColumns: {col_list}"

    st.write("### 🔹 Final Prompt Sent to API")
    st.code(final_prompt)

    if st.button("Run Query"):
        with st.spinner("Contacting API..."):

            payload = {
                "prompt": final_prompt,
                "max_tokens": 200
            }

            try:
                response = requests.post(API_URL, json=payload, verify=False)
                st.write("Status:", response.status_code)

                resp_json = response.json()
                st.write("Raw Response:", resp_json)

                model_output = resp_json.get("response", "") if isinstance(resp_json, dict) else str(resp_json)

                extracted_code = extract_code(model_output)

                st.subheader("🔹 Extracted Code")
                st.code(extracted_code, language="python")

                st.subheader("🔹 Output")

                success, out = safe_execute(extracted_code, df)

                if success:
                    if isinstance(out, pd.DataFrame):
                        st.dataframe(out)
                    elif isinstance(out, pd.Series):
                        st.dataframe(out.reset_index(name="value"))
                    else:
                        st.write(out)
                else:
                    st.error("Error executing generated code:")
                    st.code(out)

            except Exception as e:
                st.error(f"API Error: {e}")
                st.code(traceback.format_exc())