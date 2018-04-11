if __name__ == '__main__':
    import json
    from datetime import datetime
    from xmlrpc.server import SimpleXMLRPCServer
    from xmlrpc.server import SimpleXMLRPCRequestHandler
    from github import Github

    from Prediction.Linker import Linker
    import os

    from Util.github_api_methods import parse_pr_ref, parse_issue_ref

    def update_and_trim():
        global last_update, max_age_to_keep, most_recent_sha, locations, linkers
        for project in models:
            links = linkers[project].update_from_github(last_update)
            linkers[project].update_from_local_git(locations[projects], most_recent_sha)
            for link in links:
                linkers[project].update_truth(link)
            linkers[project].forget_older_than(max_age_to_keep)
            linkers[project].trim_truth()

    models = list()
    linkers = dict()
    locations = dict()
    git_locations = dict()
    last_update = dict()
    most_recent_sha = dict()

    # Restrict to a particular path.
    class RequestHandler(SimpleXMLRPCRequestHandler):
        rpc_paths = ('/RPC2',)


    # Create server
    with SimpleXMLRPCServer(("localhost", 8000), requestHandler=RequestHandler, allow_none=True) as server:
        with open(os.path.join(os.curdir, 'config.json')) as f:
            config = json.loads(f.read())

        for location in config['locations']:
            models.append(os.path.basename(location))
            locations[os.path.basename(location)] = location
            git_locations[os.path.basename(location)] = config['git_locations'][config['locations'].index(location)]
        max_age_to_keep = config['max_age_to_keep']

        for model in models:
            linkers[model] = Linker.load_from_disk(locations[model])
            most_recent_sha[model] = linkers[model].most_recent_sha()
            last_update[model] = linkers[model].most_recent_timestamp()

        server.register_introspection_functions()

        gh = Github()
        # Get lazy references to the projects on GitHub, will use to keep local models up to date
        projects = {model: gh.get_repo(model.replace('_', '/')) for model in models}

        # Register an instance; all the methods of the instance are
        # published as XML-RPC methods
        class PredictionFunctions:
            def predict_issue(self, project, issue_id):
                try:
                    issue_ref = projects[project].get_issue(int(issue_id))
                    issue = parse_issue_ref(issue_ref)
                    _, suggestions = linkers[project].update_and_predict((issue, issue))
                    suggestions = [(id_[len('issue_'):], '#%s: %s'
                                    % (id_[len('issue_'):], linkers[project].id_to_title(id_)), prop)
                                   for id_, prop in suggestions]
                    return list(suggestions)
                except KeyError:
                    return None

            def predict_pr(self, project, pr_id):
                try:
                    pr_ref = projects[project].get_pull(int(pr_id))
                    pr = parse_pr_ref(pr_ref, project)
                    _, suggestions = linkers[project].update_and_predict((pr, pr))
                    suggestions = [(id_[len('issue_'):], '#%s: %s'
                                    % (id_[len('issue_'):], linkers[project].id_to_title(id_)), prop)
                                   for id_, prop in suggestions]
                    return list(suggestions)
                except KeyError:
                    return None

            def trigger_model_updates(self):
                for model in models:
                    linkers[model].update_from_github(gh, last_update[model])
                    last_update[model] = datetime.now()
                    linkers[model].update_from_local_git(git_locations[model], most_recent_sha[model])
                    linkers[model].forget_older_than(max_age_to_keep)
                    linkers[model].persist_to_disk(locations[model])

            def record_link(self, project, issue_id, pr_id):
                linkers[project].update_truth((issue_id, pr_id))


        server.register_instance(PredictionFunctions())
        print('Loading done, entering serve loop')
        # Run the server's main loop
        server.serve_forever()
