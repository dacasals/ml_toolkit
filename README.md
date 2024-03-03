# ml_toolkit
Filling some of my ML gaps with code :)

### Streamlit app
Inside /app folder there is an Streamlit app with some ML implemented for:
- Sample size estimation for a two sided AB Test XP.
- An app to run XP analisys for a two sided A/B Test, you can fill some parameters and upload youd Raw XP results .csv
  - requires a group column
  - a randomization unit column to remove duplicates
  - a column **converted** with 1,0 values(it is fixed to apply a CVR example, it will be generalized lated)
