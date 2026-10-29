import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Loan Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Loan Approval Analysis Dashboard")
st.write("Explore loan applications, applicant income, CIBIL scores, and approval patterns.")

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

df["loan_status"] = df["loan_status"].str.title()
df["education"] = df["education"].str.title()

# Sidebar filters
st.sidebar.header("Filters")

statuses = st.sidebar.multiselect(
    "Loan Status",
    options=sorted(df["loan_status"].dropna().unique()),
    default=sorted(df["loan_status"].dropna().unique())
)

education_options = st.sidebar.multiselect(
    "Education",
    options=sorted(df["education"].dropna().unique()),
    default=sorted(df["education"].dropna().unique())
)

filtered_df = df[
    df["loan_status"].isin(statuses) &
    df["education"].isin(education_options)
]

if filtered_df.empty:
    st.warning("No applications match these filters.")
    st.stop()

# Summary cards
total = len(filtered_df)
approved = (filtered_df["loan_status"] == "Approved").sum()
rejected = (filtered_df["loan_status"] == "Rejected").sum()
approval_rate = approved / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Applications", f"{total:,}")
c2.metric("Approved", f"{approved:,}")
c3.metric("Rejected", f"{rejected:,}")
c4.metric("Approval Rate", f"{approval_rate:.2f}%")

st.divider()

# Visualization 1: Loan status
st.subheader("1. Loan Approval vs Rejection")
fig, ax = plt.subplots()
filtered_df["loan_status"].value_counts().plot(
    kind="bar", ax=ax
)
ax.set_xlabel("Loan Status")
ax.set_ylabel("Number of Applications")
ax.tick_params(axis="x", rotation=0)
st.pyplot(fig)
plt.close(fig)

# Visualization 2: Income distribution
st.subheader("2. Annual Income Distribution")
fig, ax = plt.subplots()
ax.hist(filtered_df["income_annum"] / 100000, bins=20, edgecolor="black")
ax.set_xlabel("Annual Income (₹ Lakhs)")
ax.set_ylabel("Number of Applicants")
st.pyplot(fig)
plt.close(fig)

# Visualization 3: CIBIL score by loan status
st.subheader("3. CIBIL Score by Loan Status")
fig, ax = plt.subplots()
groups = [
    filtered_df.loc[
        filtered_df["loan_status"] == status, "cibil_score"
    ].dropna()
    for status in ["Approved", "Rejected"]
]
labels = [
    status for status, group in zip(["Approved", "Rejected"], groups)
    if not group.empty
]
groups = [group for group in groups if not group.empty]

if groups:
    ax.boxplot(groups, tick_labels=labels)
    ax.set_ylabel("CIBIL Score")
    st.pyplot(fig)
else:
    st.info("No CIBIL data available for the selected statuses.")
plt.close(fig)

# Visualization 4: Loan status by education
st.subheader("4. Loan Status by Education")
education_status = pd.crosstab(
    filtered_df["education"], filtered_df["loan_status"]
)
fig, ax = plt.subplots()
education_status.plot(kind="bar", ax=ax)
ax.set_xlabel("Education")
ax.set_ylabel("Number of Applications")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Loan Status")
st.pyplot(fig)
plt.close(fig)

# Visualization 5: Numeric correlation heatmap
st.subheader("5. Correlation Between Numeric Variables")
numeric_df = filtered_df.select_dtypes(include="number").drop(
    columns=["loan_id"], errors="ignore"
)
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 6))
image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7)

fig.colorbar(image, ax=ax, label="Correlation")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Dataset explorer
with st.expander("View Filtered Dataset"):
    st.dataframe(filtered_df, use_container_width=True)

st.download_button(
    "Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_loan_data.csv",
    mime="text/csv"
)
