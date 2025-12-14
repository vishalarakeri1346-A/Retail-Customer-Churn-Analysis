-- Total customers
SELECT COUNT(CustomerID) AS TotalCustomers FROM customers;

-- Churn rate
SELECT ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS ChurnRatePercentage
FROM customers;

-- Average Monthly Revenue
SELECT ROUND(AVG(MonthlyCharges), 2) AS AverageMonthlyRevenue FROM customers;

-- Count by membership plan
SELECT Membership, COUNT(*) AS CustomerCount FROM customers GROUP BY Membership ORDER BY CustomerCount DESC;

-- Churn by membership plan
SELECT Membership, COUNT(*) AS TotalCustomers,
    SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) AS Churned,
    ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS ChurnRate
FROM customers GROUP BY Membership ORDER BY ChurnRate DESC;

-- Revenue lost due to churn
SELECT SUM(MonthlyCharges) AS MonthlyRevenueLost, SUM(TotalCharges) AS TotalRevenueLost FROM customers WHERE Churn = 'Yes';

-- Tenure-based churn
SELECT CASE WHEN Tenure < 6 THEN '0-6' WHEN Tenure BETWEEN 6 AND 12 THEN '6-12' WHEN Tenure BETWEEN 12 AND 24 THEN '12-24' ELSE '24+' END AS TenureGroup,
    COUNT(*) AS TotalCustomers,
    SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) AS Churned,
    ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS ChurnRate
FROM customers GROUP BY TenureGroup ORDER BY ChurnRate DESC;
