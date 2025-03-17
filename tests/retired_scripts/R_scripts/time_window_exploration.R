#time_window_exploration.R

data("identified_individuals")

groups_by_individual <- get_associations_points_tw(identified_individuals, 
                                                   time_window = 30, 
                                                   which_days = NULL, 
                                                   which_locations = NULL)


