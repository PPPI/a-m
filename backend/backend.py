from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from github import Github

from Prediction.Linker import Linker
import os

from Util.github_api_methods import parse_pr_ref

models = ['PhilJay_MPAndroidChart', 'google_guava']
linkers = dict()


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


# Create server
with SimpleXMLRPCServer(("localhost", 8000), requestHandler=RequestHandler) as server:
    for model in models:
        linkers[model] = Linker.load_from_disk(os.path.join(os.curdir, 'models', model))
    server.register_introspection_functions()
    gh = Github()
    # Get lazy references to the projects on GitHub, will use to keep local models up to date
    # TODO: Forgetting acquired information
    projects = {model: gh.get_repo(model.replace('_', '/')) for model in models}

    # Register an instance; all the methods of the instance are
    # published as XML-RPC methods
    class PredictionFunctions:
        # TODO: Create issue symmetric case
        # def predict_issue(self, project, issue_id):
        #     try:
        #         issue_ref = projects[project].get_issue(issue_id)
        #         issue = parse_issue_ref(issue_ref)
        #
        #         return suggestions
        #     except KeyError:
        #         return None

        def predict_pr(self, project, pr_id):
            try:
                pr_ref = projects[project].get_pull(int(pr_id))
                pr = parse_pr_ref(pr_ref, project)
                _, suggestions = linkers[project].update_and_predict((pr, pr))
                return list(suggestions)
            except KeyError:
                return None


    server.register_instance(PredictionFunctions())
    print('Loading done, entering serve loop')
    # Run the server's main loop
    server.serve_forever()
