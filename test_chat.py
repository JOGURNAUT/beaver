"""Minimal chat_input isolation test. Run: streamlit run test_chat.py"""
import streamlit as st

st.title("Chat Input Test")
st.write("Type something below and press Enter. Check terminal.")

text = st.chat_input("type here")
print(f"[TEST] script rerun. text={text!r}", flush=True)

if text:
    st.success(f"GOT: {text}")
    print(f"[TEST] CAPTURED: {text!r}", flush=True)
