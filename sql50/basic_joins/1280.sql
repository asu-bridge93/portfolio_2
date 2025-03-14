SELECT s.student_id, s.student_name, sub.subject_name, 
       COALESCE(COUNT(e.subject_name), 0) AS attended_exams
FROM Students s
CROSS JOIN Subjects sub -- 全通り生成
LEFT JOIN Examinations e
    ON s.student_id = e.student_id AND sub.subject_name = e.subject_name -- Examinations にあるもののみを残す
GROUP BY s.student_id, s.student_name, sub.subject_name -- 組み合わせでグループ化
ORDER BY s.student_id, sub.subject_name;
