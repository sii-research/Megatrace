#!/bin/bash

# Venus Monitoring Helm Chart Installation Script
# This script handles both first-time installation and upgrades

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
NAMESPACE="monitoring"
RELEASE_NAME="venus-monitoring"
VALUES_FILE=""
DRY_RUN=false
UPGRADE=false

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Venus Monitoring Installation  ${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -n, --namespace NAME     Kubernetes namespace (default: monitoring)
    -r, --release NAME       Helm release name (default: venus-monitoring)
    -f, --values FILE        Custom values file
    -d, --dry-run           Dry run mode (show what would be installed)
    -u, --upgrade           Upgrade existing installation
    -h, --help              Show this help message

Examples:
    # First-time installation
    $0

    # Install with custom namespace
    $0 -n my-monitoring

    # Install with custom values file
    $0 -f values-production.yaml

    # Upgrade existing installation
    $0 -u

    # Dry run to see what would be installed
    $0 -d

EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        print_error "Helm is not installed. Please install Helm first."
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    print_info "Prerequisites check passed!"
}

# Function to add Prometheus community repository
add_prometheus_repo() {
    print_info "Adding Prometheus community repository..."
    
    if ! helm repo list | grep -q "prometheus-community"; then
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        print_info "Added Prometheus community repository"
    else
        print_info "Prometheus community repository already exists"
    fi
    
    print_info "Updating Helm repositories..."
    helm repo update
}

# Function to create namespace
create_namespace() {
    print_info "Creating namespace: $NAMESPACE"
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl create namespace "$NAMESPACE"
        print_info "Namespace $NAMESPACE created"
    else
        print_info "Namespace $NAMESPACE already exists"
    fi
}

# Function to install or upgrade
install_or_upgrade() {
    local chart_path="."
    local helm_cmd=""
    
    if [ "$UPGRADE" = true ]; then
        print_info "Upgrading existing installation..."
        helm_cmd="helm upgrade --install"
    else
        print_info "Installing Venus Monitoring..."
        helm_cmd="helm install"
    fi
    
    # Build helm command
    local cmd="$helm_cmd $RELEASE_NAME $chart_path"
    cmd="$cmd --namespace $NAMESPACE"
    
    if [ -n "$VALUES_FILE" ]; then
        cmd="$cmd --values $VALUES_FILE"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        cmd="$cmd --dry-run --debug"
    fi
    
    print_info "Executing: $cmd"
    
    if eval "$cmd"; then
        if [ "$DRY_RUN" = true ]; then
            print_info "Dry run completed successfully"
        else
            print_info "Installation completed successfully!"
        fi
    else
        print_error "Installation failed!"
        exit 1
    fi
}

# Function to show post-installation information
show_post_install_info() {
    if [ "$DRY_RUN" = false ]; then
        print_info "Post-installation information:"
        echo ""
        echo "Namespace: $NAMESPACE"
        echo "Release: $RELEASE_NAME"
        echo ""
        echo "To check the status:"
        echo "  kubectl get pods -n $NAMESPACE"
        echo "  helm status $RELEASE_NAME -n $NAMESPACE"
        echo ""
        echo "To access Grafana:"
        echo "  kubectl port-forward -n $NAMESPACE svc/$RELEASE_NAME-grafana 3000:80"
        echo ""
        echo "To access Prometheus:"
        echo "  kubectl port-forward -n $NAMESPACE svc/$RELEASE_NAME-prometheus-server 9090:80"
        echo ""
        echo "To uninstall:"
        echo "  helm uninstall $RELEASE_NAME -n $NAMESPACE"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--release)
            RELEASE_NAME="$2"
            shift 2
            ;;
        -f|--values)
            VALUES_FILE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -u|--upgrade)
            UPGRADE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header
    
    # Check prerequisites
    check_prerequisites
    
    # Add Prometheus repository
    add_prometheus_repo
    
    # Create namespace
    create_namespace
    
    # Install or upgrade
    install_or_upgrade
    
    # Show post-installation information
    show_post_install_info
    
    print_info "Script completed successfully!"
}

# Run main function
main "$@"