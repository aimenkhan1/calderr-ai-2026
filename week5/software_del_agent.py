"""
SOFTWARE DELIVERY SIMULATION
===========================================================
Flow:
  PM Agent decomposes requirements
      -> Backend Agent + Frontend Agent build IN PARALLEL
      -> QA Agent tests the combined result
      -> PM Agent synthesizes a final release report (GO / NO-GO)
"""

from __future__ import annotations
from enum import Enum
from typing import List
from pydantic import BaseModel, Field
import threading
import random
import time



class RequirementSpec(BaseModel):

    sender: str          
    recipient: str      
    component: str
    description: str     


class BuildStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class BuildResult(BaseModel):
    sender: str            
    recipient: str        
    component: str
    status: BuildStatus
    files_changed: int
    notes: str


class TestReport(BaseModel):

    sender: str              
    recipient: str           
    tests_run: int
    tests_passed: int
    issues: List[str]    
    passed: bool            


class ReleaseDecision(str, Enum):
    GO = "go"
    NO_GO = "no_go"


class ReleaseReport(BaseModel):

    decision: ReleaseDecision
    summary: str
    build_results: List[BuildResult]
    test_report: TestReport
    blocking_issues: List[str]


#frontend and backend running parallelly on separate threads, then QA tests the combined result, then PM synthesizes a final report

class BackendAgent:
    name = "Backend"

    def build(self, spec: RequirementSpec) -> BuildResult:
        time.sleep(random.uniform(0.3, 0.6))  
        failed = random.random() < 0.15
        return BuildResult(
            sender=self.name,
            recipient="PM",
            component=spec.component,
            status=BuildStatus.FAILED if failed else BuildStatus.SUCCESS,
            files_changed=random.randint(3, 12),
            notes=(
                "Build failed: unresolved dependency in the API layer."
                if failed else
                f"Implemented: {spec.description}"
            ),
        )


class FrontendAgent:
    name = "Frontend"

    def build(self, spec: RequirementSpec) -> BuildResult:
        time.sleep(random.uniform(0.3, 0.6))
        failed = random.random() < 0.15
        return BuildResult(
            sender=self.name,
            recipient="PM",
            component=spec.component,
            status=BuildStatus.FAILED if failed else BuildStatus.SUCCESS,
            files_changed=random.randint(2, 8),
            notes=(
                "Build failed: component did not compile."
                if failed else
                f"Implemented: {spec.description}"
            ),
        )


#qa agent tests the combined result of backend and frontend builds

class QAAgent:
    name = "QA"

    def test(self, build_results: List[BuildResult]) -> TestReport:
        time.sleep(0.3)

        # if either build already failed outright, QA can't even run properly
        any_build_failed = any(r.status == BuildStatus.FAILED for r in build_results)

        tests_run = 20
        issues: List[str] = []

        if any_build_failed:
            failed_components = [r.component for r in build_results if r.status == BuildStatus.FAILED]
            issues.append(f"Cannot fully test - build failed for: {', '.join(failed_components)}")
            tests_passed = random.randint(0, 5)   # only a handful of unrelated tests can even run
        else:
            # even a "successful" build can still fail some tests - this is normal
            tests_passed = random.randint(15, 20)
            if tests_passed < tests_run:
                issues.append(f"{tests_run - tests_passed} test(s) failed - see logs for details.")

        return TestReport(
            sender=self.name,
            recipient="PM",
            tests_run=tests_run,
            tests_passed=tests_passed,
            issues=issues,
            passed=(tests_passed == tests_run) and not any_build_failed,
        )


# pm agent decomposes requirements, runs builds in parallel, and synthesizes final report

class PMAgent:
    name = "PM"

    def decompose(self, requirements: str) -> tuple[RequirementSpec, RequirementSpec]:
    
        backend_spec = RequirementSpec(
            sender=self.name,
            recipient="Backend",
            component="backend",
            description=f"Build the API and data layer for: {requirements}",
        )
        frontend_spec = RequirementSpec(
            sender=self.name,
            recipient="Frontend",
            component="frontend",
            description=f"Build the UI for: {requirements}",
        )
        return backend_spec, frontend_spec

#Runs Backend.build() and Frontend.build() on SEPARATE THREAD at the same time, since neither depends on the other's output. This is genuine parallelism
    def run_parallel_builds(self, backend_agent: BackendAgent, frontend_agent: FrontendAgent,
                             backend_spec: RequirementSpec, frontend_spec: RequirementSpec) -> List[BuildResult]:

        results: dict[str, BuildResult] = {}

        def run_backend():
            results["backend"] = backend_agent.build(backend_spec)

        def run_frontend():
            results["frontend"] = frontend_agent.build(frontend_spec)

        t1 = threading.Thread(target=run_backend)
        t2 = threading.Thread(target=run_frontend)

        start = time.time()
        t1.start()
        t2.start()
        t1.join()   # wait for backend thread to finish
        t2.join()   # wait for frontend thread to finish
        elapsed = time.time() - start

        print(f"  [PM] Both builds completed in {elapsed:.2f}s (running in parallel, not sequentially)")

        return [results["backend"], results["frontend"]]

    def synthesize_release_report(self, build_results: List[BuildResult], test_report: TestReport) -> ReleaseReport:

        blocking_issues: List[str] = []

        failed_builds = [r for r in build_results if r.status == BuildStatus.FAILED]
        for r in failed_builds:
            blocking_issues.append(f"{r.component} build failed: {r.notes}")

        if not test_report.passed:
            blocking_issues.extend(test_report.issues)

        decision = ReleaseDecision.NO_GO if blocking_issues else ReleaseDecision.GO

        summary = (
            f"{len(build_results)} components built "
            f"({sum(1 for r in build_results if r.status == BuildStatus.SUCCESS)} succeeded, "
            f"{len(failed_builds)} failed). "
            f"QA: {test_report.tests_passed}/{test_report.tests_run} tests passed. "
            f"Decision: {decision.value.upper()}."
        )

        return ReleaseReport(
            decision=decision,
            summary=summary,
            build_results=build_results,
            test_report=test_report,
            blocking_issues=blocking_issues,
        )


#pipeline

def run_delivery_pipeline(requirements: str) -> ReleaseReport:
    pm = PMAgent()
    backend_agent = BackendAgent()
    frontend_agent = FrontendAgent()
    qa_agent = QAAgent()

    print(f"--- Delivering: '{requirements}' ---\n")

    # Step 1: PM decomposes requirements into 2 scoped specs
    backend_spec, frontend_spec = pm.decompose(requirements)
    print(f"  [PM] Decomposed into:\n"
          f"    - Backend: {backend_spec.description}\n"
          f"    - Frontend: {frontend_spec.description}")

    # Step 2: Backend + Frontend build IN PARALLEL
    build_results = pm.run_parallel_builds(backend_agent, frontend_agent, backend_spec, frontend_spec)
    for r in build_results:
        print(f"  [{r.sender}] {r.status.value.upper()} - {r.notes} ({r.files_changed} files changed)")

    # Step 3: QA tests the combined result
    test_report = qa_agent.test(build_results)
    print(f"  [QA] {test_report.tests_passed}/{test_report.tests_run} tests passed. "
          f"Issues: {test_report.issues if test_report.issues else 'none'}")

    # Step 4: PM synthesizes the final release report
    report = pm.synthesize_release_report(build_results, test_report)

    print("\n===================================\n")
    print("RELEASE REPORT")
    print("\n===================================\n")

    print(report.model_dump_json(indent=2))

    return report


if __name__ == "__main__":
    run_delivery_pipeline("User login with email and password")