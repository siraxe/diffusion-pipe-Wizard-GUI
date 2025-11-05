#!/usr/bin/env python3
"""
Helper script to detect SimplePod.ai or other tunnel service URLs
"""
import os
import subprocess
import sys
import json
import re
from urllib.parse import urlparse

def check_environment_variables():
    """Check for tunnel URLs in environment variables"""
    print("=== Checking Environment Variables ===")

    tunnel_vars = [
        "SIMPLEPOD_URL", "TUNNEL_URL", "PUBLIC_URL", "EXTERNAL_URL",
        "NGROK_URL", "CLOUDFLARED_URL", "PAGEKITE_URL"
    ]

    found_urls = []
    for var in tunnel_vars:
        url = os.getenv(var)
        if url:
            print(f"[+] {var}: {url}")
            found_urls.append(url)

    # Also check for any variable containing "url" or "host"
    env_vars = dict(os.environ)
    for key, value in env_vars.items():
        if any(keyword in key.lower() for keyword in ['url', 'host', 'tunnel']) and '://' in str(value):
            print(f"[+] {key}: {value}")
            found_urls.append(str(value))

    return found_urls

def check_ngrok():
    """Check for ngrok tunnels"""
    print("\n=== Checking ngrok ===")
    try:
        result = subprocess.run(["curl", "-s", "http://localhost:4040/api/tunnels"],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for tunnel in data.get("tunnels", []):
                if tunnel.get("proto") == "http":
                    url = tunnel["public_url"]
                    print(f"[+] ngrok tunnel found: {url}")
                    return url
        else:
            print("[-] ngrok API not responding")
    except Exception as e:
        print(f"[-] ngrok check failed: {e}")

    return None

def check_cloudflared():
    """Check for cloudflare tunnels"""
    print("\n=== Checking Cloudflare Tunnel ===")
    try:
        # Try common metrics endpoints
        endpoints = ["http://localhost:42591/metrics", "http://localhost:6123/metrics"]
        for endpoint in endpoints:
            try:
                result = subprocess.run(["curl", "-s", endpoint],
                                      capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    # Parse cloudflare tunnel metrics for URL
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'tunnel_host' in line:
                            # Extract hostname from metrics
                            host = line.split('tunnel_host="')[1].split('"')[0]
                            url = f"https://{host}"
                            print(f"[+] Cloudflare tunnel found: {url}")
                            return url
            except:
                continue

        print("[-] Cloudflare tunnel not found")
    except Exception as e:
        print(f"[-] Cloudflare tunnel check failed: {e}")

    return None

def check_network_connections():
    """Check for unusual network connections that might be tunnels"""
    print("\n=== Checking Network Connections ===")
    try:
        # Check for processes listening on common tunnel ports
        tunnel_ports = [39004, 4040, 42591, 8080, 8443, 3000]

        # Try different commands based on what's available
        commands = [
            ["netstat", "-tuln"],
            ["ss", "-tuln"],
            ["lsof", "-i"]
        ]

        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if any(str(port) in line for port in tunnel_ports):
                            print(f"[+] Found connection: {line.strip()}")
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Command {cmd[0]} failed: {e}")

    except Exception as e:
        print(f"[-] Network check failed: {e}")

def check_running_processes():
    """Check for running tunnel processes"""
    print("\n=== Checking Running Processes ===")
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            tunnel_keywords = ['simplepod', 'ngrok', 'cloudflared', 'tunnel', 'frp', 'pagekite']

            for line in lines:
                if any(keyword in line.lower() for keyword in tunnel_keywords):
                    print(f"[+] Found process: {line.strip()}")
    except Exception as e:
        print(f"[-] Process check failed: {e}")

def main():
    print("SimplePod.ai URL Detection Script")
    print("=" * 40)

    urls = []

    # Check environment variables
    env_urls = check_environment_variables()
    urls.extend(env_urls)

    # Check ngrok
    ngrok_url = check_ngrok()
    if ngrok_url:
        urls.append(ngrok_url)

    # Check cloudflare
    cloudflare_url = check_cloudflared()
    if cloudflare_url:
        urls.append(cloudflare_url)

    # Check network connections
    check_network_connections()

    # Check running processes
    check_running_processes()

    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY:")

    if urls:
        print("Found URLs:")
        for i, url in enumerate(urls, 1):
            parsed = urlparse(url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == 'https' else 80)
            print(f"  {i}. {url}")
            print(f"     Host: {host}, Port: {port}")

        print("\nTo use these URLs with your app:")
        print(f"export SIMPLEPOD_URL='{urls[0]}'")
        print("python flet_app/flet_app.py")

    else:
        print("No automatic tunnel detection found.")
        print("\nManual setup options:")
        print("1. Set environment variable:")
        print("   export SIMPLEPOD_URL='http://217.171.200.22:39004'")
        print("2. Or set in your app:")
        print("   export FLET_SERVER_HOST='217.171.200.22'")
        print("   export FLET_SERVER_PORT='39004'")

if __name__ == "__main__":
    main()