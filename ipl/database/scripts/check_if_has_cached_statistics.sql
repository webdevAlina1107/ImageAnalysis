SELECT exists(select 1 from statistic_info WHERE standard_deviation IS NOT NULL AND cloud_rate IS NOT NULL AND
index_weighted_avg IS NOT NULL AND confidence_interval_lower IS NOT NULL AND confidence_interval_upper IS NOT NULL
AND image_id = ? LIMIT 1)