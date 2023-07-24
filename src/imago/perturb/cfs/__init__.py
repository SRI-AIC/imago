class CounterFactual:
    def __init__(self, query_datum, cf_datum, cf_Z, odiff):
        self.query_datum = query_datum
        self.cf_datum = cf_datum
        self.cf_Z = cf_Z
        self.odiff = odiff


def get_cfs(query_datum, perturb, domain, inst_ds, **kwargs):
    """ Contract interface for each module to implement for
    returning a dictionary of counterfactuals."""
    pass


METHOD = "Method"
ODIFF = "ODiff"
MIN_INST_ODIFF = "MinInstODiff"
ANOM = "Anom"
NUN = "NUN"
CF_MET = "CFMet"
INTERP1 = "InterpPt1"
INTERP1_ADDREAL = "InterpPt1 RealAdj"
INTERP2 = "InterpPt2"
GRAD = "Gradient"
GRAD_NOREAL = "Gradient NoAdj"
