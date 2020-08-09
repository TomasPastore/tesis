class Event():
    def __init__(self, info):
        '''
        :param info: Is a dictionary with the event properties
        '''
        self.info = info

    def reset_preds(self, models_to_run):
        self.info['prediction'] = {m: [0, 0] for m in models_to_run}
        self.info['proba'] = {m: 0 for m in models_to_run}
