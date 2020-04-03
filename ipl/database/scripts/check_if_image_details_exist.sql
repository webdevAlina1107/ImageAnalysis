select exists(select 1 from image where field_id=? and revision=? and capture_date=?
  and mysterious_date=? and capture_satellite=? limit 1)