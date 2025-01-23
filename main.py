# ייבוא ספריות
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

# טעינת הנתונים לתוך DataFrame
data = load_diabetes(as_frame=True)
df = data['data']
df['target'] = data['target']

# הצגת מידע על הנתונים
print(df.info())
print(df.head())

# חלק 1: היסטוגרמות
# BMI
plt.hist(df['bmi'], bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.title("Distribution of BMI")
plt.xlabel("BMI (Normalized)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("bmi_histogram.png")
plt.show()

# Blood Pressure (BP)
plt.hist(df['bp'], bins=20, color='green', alpha=0.7, edgecolor='black')
plt.title("Distribution of Blood Pressure")
plt.xlabel("Blood Pressure (Normalized)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("bp_histogram.png")
plt.show()

# Sex
plt.hist(df['sex'], bins=20, color='purple', alpha=0.7, edgecolor='black')
plt.title("Distribution of Sex")
plt.xlabel("Sex (Normalized)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("sex_histogram.png")
plt.show()

# חלק 2: גרפי פיזור
# BMI מול Target
plt.scatter(df['bmi'], df['target'], alpha=0.5, color="blue", edgecolor="black")
plt.title("BMI vs Diabetes Progression")
plt.xlabel("BMI (Normalized)")
plt.ylabel("Diabetes Progression (Target)")
plt.grid(True)
plt.tight_layout()
plt.savefig("bmi_vs_target.png")
plt.show()

# BP מול Target
plt.scatter(df['bp'], df['target'], alpha=0.5, color="green", edgecolor="black")
plt.title("Blood Pressure vs Diabetes Progression")
plt.xlabel("Blood Pressure (Normalized)")
plt.ylabel("Diabetes Progression (Target)")
plt.grid(True)
plt.tight_layout()
plt.savefig("bp_vs_target.png")
plt.show()

# Sex מול Target
plt.scatter(df['sex'], df['target'], alpha=0.5, color="purple", edgecolor="black")
plt.title("Sex vs Diabetes Progression")
plt.xlabel("Sex (Normalized)")
plt.ylabel("Diabetes Progression (Target)")
plt.grid(True)
plt.tight_layout()
plt.savefig("sex_vs_target.png")
plt.show()

# חלק 3: גרף המתאמים (Bar-Connected Graph)
# חישוב מתאמים עם Target
correlations = df.corr()['target'].sort_values(ascending=False)

# יצירת גרף עמודות מחוברות
plt.plot(correlations.index, correlations.values, marker='o', linestyle='-', color='blue', label='Correlation with Target')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title("Correlation with Target (Bar-Connected Graph)", fontsize=16)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Correlation Value", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend()
plt.tight_layout()
plt.savefig("correlation_with_target.png")
plt.show()
