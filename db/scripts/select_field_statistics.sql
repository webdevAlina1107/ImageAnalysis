select statistic_info.cloud_rate, statistic_info.index_weighted_avg, statistic_info.standard_deviation,
 statistic_info.confidence_interval_lower, statistic_info.confidence_interval_upper from statistic_info
inner join image on (statistic_info.image_id = image.image_id) where image.field_id = ?
and image.data between ? and ? and statistic_info.cloud_rate < ? order by image.data asc