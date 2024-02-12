Class Structure:
(A) Detectors: Manage the data from one specific metric or, "detector"
(B) DetectorLayers: Manage a set of detectors that are all at the same "level"
(C) HierarchalManager: Manage all of the detector layers
(D) LayerGenerator: Takes in a DetectorLater and creates a "higher-level" detector from grouping and clustering.

Distribution Change & Event Detection
1) Data is read in from .csv files and loaded into one detector object
2) For each detector, the find_unique_events() function is called in Detector class
3) The ENTIRE data is divided into overlapping chunks and each chunk is passed into find_events_via_ts() function: 
    (*) The amount of chunks that the ENTIRE data is split into can be specified
    (*) Every chunk overlaps with half of the previous and later chunk
    (*) Depending on the input size, you can change the chunk size to get a good gauge on events 
        (**) Increasing the amount of chunks will decrease the amount of events
        (**) However, there is not real way to to clump metrics by chunks, as far as I can see.
    (A) find_events_via_ts() finds the top 3.5% most dissimilar points in each chunk 
        (*) You can change the percentage but you want enough points to handle small 
        (*) This number is what I used to test and it works fine really for anything less than 5%.
    (B) Dissimilarity: This measures the difference in mean before and after a specific point
    (C) For every point in this chunk, calculate the mean before and after this specific point and and find the dissimilarity
    (D) Dissimilarity is calculated with squared_hellinger_distance(), or 
        (i) 1 - math.e ** (-rho * ((math.sqrt(lambda_p) - math.sqrt(lambda_q)) ** 2) / 2)
        (ii) The lambda's are the two means and rho is a trained variable
        (iii) Rho is trained so the saturation of the squared_hellinger_distance() is less than 0.1%
        (iv) SATURATION: A calculuation of the squared_hellinger_distance() function is SATURATED if returns 1
        (v) Threshold: a score above this threshold indicates that this points could be a possible event, also a trainable parameter
    (E) From this, we will get a set of possible events for step 4)
4) Create "range compartments": small, NON-OVERLAPPING ranges, or boxes based on x-coordinates, similar to the chunks in 3) but non-overlapping
    (A) We want to only choose points from compartments that have more than one supposed "event" in them
        (i) We have the make the compartments small enough to avoid having two events in one compartment but large enough to put events of similar x-coords in the compartment
    (B) Put every possible event into a compartment, compartment is based on the data's x-coord.
5) From all the events in compartments created in step 4, we filter out points on the following two principles:
    (A) Throw out all events from chunks that do not see a large change in value    
        (*) An event should see a distribution change, which should see some significant change in y-value
        (*) Sometimes you see massive dips without a return, so this is an issue
        (i) Currently, large is defined as 60% of the different in the entire data's y-max ans y-min
            (*) Again this is just a number I found worked best, I don't have no real other reason
    (B) Only take points that are in a compartment that contains multiple points
        (i) Because of the overlapping chunks, every important event should be in a compartment with at least 2 points
        (ii) Compartments with 1 or less most likely don't contain any interesting events and are just a byproduct of trying to find possible events without knowing if there ARE any events
6) From this you get a set of events to plot
    (A) You may need to adjust the CHUNKS parameter if you are not happy with the amount of points you get
    (B) I suspect this has to do with the data itself, but will check out if X-coords have anything to do with it


Methods:
1) Start off your data collection
2) Keep growing the data you analyze until it reaches a desired size: ~400 points
    A) Then, just take the most recent 400 points
3) Try and find anomalies in this region
4) In each detector keep track of the x-coords of points every time you run step 3
5) To verify true anomalies, if a region shows up more than once, it is called an anomaly


1) Looking into hierarchal monitoring system
    A) Originally thought it was the amount of data
    B) Presence of Outliers
    C) Data is not normally distributed -> straight lines and cycles
        D) A lot of things assoicated with not helpful data
    E) Still think you should have it
        *) Cluster similar metrics into one
        *) View known relationships
        *) group without cluster?
2) Caching metrics -> DCGM
    A) job 
3) Time complexity -> Reduced
    A) Stuff is taking too long to run
    B) Spends too much time on non-interesting data
    C) Max Iterations

