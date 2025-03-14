SELECT w.id AS Id FROM Weather w
JOIN Weather p
ON DATE_ADD(p.recordDate, INTERVAL 1 DAY) = w.recordDate
WHERE w.temperature > p.temperature;
