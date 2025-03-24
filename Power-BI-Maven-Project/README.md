
# Maven Market Sales Dashboard 📊

This Power BI project showcases an end-to-end business intelligence solution built on a fictional retail dataset provided by Maven Analytics. The goal was to explore sales performance across products, regions, and time while enabling decision-makers to track KPIs and uncover key trends.

## 🚀 Project Overview

This dashboard was built to analyze the sales and return activity of a fictional retail chain called **Maven Market**, covering the years 1997 and 1998. It helps answer important business questions such as:

- What were the total sales, profit, and return rate across regions?
- Which product categories performed the best?
- Are there any seasonal trends or spikes in returns?
- Which stores or regions are underperforming?

## 📁 Project Structure

```
📂 Maven Market Project
├── MavenMarket_Report.pbix                # Power BI dashboard file
├── Maven_Market.png                       # The icon of Maven Market 
├── Maven_Market_Dashboard.png  	         # Screenshot of the final dashboard
├── Power BI Demo Video.mp4                # Walkthrough demo video
├── Maven Transactions/                    # Raw transaction data for 1997 & 1998
├── Maven+Market+CSV+Files/                # Supporting datasets (Products, Customers, Stores, etc.)
```

## 🔧 Key Features & Steps

- **Data Cleaning**: Combined multiple CSV files (transactions, customers, returns, etc.) and ensured consistency in date formats and fields.
- **Data Modeling**: Built a star schema data model with clear relationships between dimension and fact tables.
- **Calculated Columns & Measures**: Created DAX measures for KPIs like Total Sales, Profit Margin, Return Rate, YoY % Change, etc.
- **Interactive Dashboard**:
  - Region & store-level filtering
  - Category & product-level drilldowns
  - Monthly trend analysis with slicers
- **Visual Design**: Focused on clarity and usability. Used cards, bar charts, maps, and trend lines to convey insights.

## 🖼️ Sample Dashboard

![Dashboard Preview](Maven_Market_Dashboard.png)

## 🎬 Demo Video

A short demo video is included (`Power BI Demo Video.mp4`) to walk through the dashboard functionality and how it supports business decision-making.

## 🧠 What I Learned

- Structuring complex datasets for analytical modeling
- Using DAX for custom KPIs and conditional formatting
- Designing clean and functional business dashboards
- Best practices for data storytelling with Power BI

## 📌 Tools Used

- Power BI Desktop
- Microsoft Excel (for quick data inspection)
- DAX (Data Analysis Expressions)
- Git for version control

---

> This project was developed as part of a Power BI practice challenge by Maven Analytics. It demonstrates my ability to turn raw data into actionable business insights using visual analytics.

