SELECT (standart_deviation IS NOT NULL AND cloud_rate IS NOT NULL AND index_weighted_avg IS NOT NULL
AND confidence_interval_lower IS NOT NULL AND confidence_interval_upper IS NOT NULL) from statistic_info
WHERE image_id = ?