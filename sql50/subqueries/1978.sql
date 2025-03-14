SELECT employee_id
FROM Employees e1
WHERE e1.salary < 30000
    AND e1.manager_id NOT IN (
        SELECT e2.employee_id
        FROM Employees e2
    )
ORDER BY employee_id;
