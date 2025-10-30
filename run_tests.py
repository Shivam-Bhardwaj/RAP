#!/usr/bin/env python3
"""
Test runner and configuration for RAP-ID test suite.

Usage:
    python -m pytest tests/ -v
    python run_tests.py --unit
    python run_tests.py --integration
    python run_tests.py --all
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_unit_tests():
    """Run unit tests."""
    print("=" * 60)
    print("Running Unit Tests")
    print("=" * 60)
    return subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_uncertainty.py",
        "tests/test_losses.py",
        "tests/test_models.py",
        "-v",
        "--tb=short"
    ]).returncode


def run_integration_tests():
    """Run integration tests."""
    print("=" * 60)
    print("Running Integration Tests")
    print("=" * 60)
    return subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_integration.py",
        "-v",
        "--tb=short"
    ]).returncode


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running All Tests")
    print("=" * 60)
    return subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term"
    ]).returncode


def run_benchmarks():
    """Run performance benchmarks."""
    print("=" * 60)
    print("Running Performance Benchmarks")
    print("=" * 60)
    return subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_benchmarks.py",
        "-v",
        "--benchmark-only"
    ]).returncode


def main():
    parser = argparse.ArgumentParser(description="RAP-ID Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--benchmarks", action="store_true", help="Run benchmarks only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.unit:
        exit_code = run_unit_tests()
    elif args.integration:
        exit_code = run_integration_tests()
    elif args.benchmarks:
        exit_code = run_benchmarks()
    elif args.all:
        exit_code = run_all_tests()
    else:
        # Default: run unit tests
        exit_code = run_unit_tests()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

