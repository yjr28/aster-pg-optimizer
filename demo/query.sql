SELECT c.segment, count(*) AS orders, sum(o.amount) AS revenue
FROM customers AS c
JOIN orders AS o ON o.customer_id = c.id
WHERE c.region = 'west' AND o.status IN (1, 2, 3)
GROUP BY c.segment
ORDER BY revenue DESC;
