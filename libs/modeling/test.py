import matplotlib.pyplot as plt

# Data
years = [2015, 2017, 2020, 2021, 2022, 2023]
percentages = [3.4, 3.3, 10, 10, 13.3, 60]

# Plot
fig, ax = plt.subplots(figsize=(14, 8))

# Line plot
ax.plot(years, percentages, marker='o', linestyle='-', color='b')

# Add labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Percentage (%)')
ax.set_title('Increase in the Creation of AI Deepfake Tools (2015-2023)')
ax.set_xticks(years)
ax.set_ylim(0, 70)
ax.grid(True, linestyle='--', alpha=0.7)

# Annotate each point with the percentage
for i, (year, percentage) in enumerate(zip(years, percentages)):
    ax.annotate(
        f'{percentage}%',
        xy=(year, percentage),
        xytext=(5, 5),
        textcoords='offset points',
        ha='left',
        va='bottom',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8)
    )

# Highlight the significant increase in 2023
ax.axvspan(2022.5, 2023.5, color='red', alpha=0.2)
# ax.text(2023, 62, "Significant Increase in 2023", ha='center', va='top', fontsize=12, color='red')

plt.show()
