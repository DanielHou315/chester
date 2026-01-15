#!/usr/bin/env python
"""
Simple MNIST training script for testing chester.

Supports both MLP and CNN architectures with lazy PyTorch import.

Usage (standalone):
    python tests/train_mnist.py --model mlp --epochs 2 --batch_size 64

Usage (via chester):
    python tests/launch_mnist.py --mode local
"""
import os
import argparse


def get_device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_mlp(input_dim=784, hidden_dim=256, num_classes=10):
    """Build a simple MLP."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim // 2, num_classes),
    )


def build_cnn(num_classes=10):
    """Build a simple CNN."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    import torch
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(loader), correct / total


def run_training(variant, log_dir, exp_name):
    """
    Main training function called by chester.

    Args:
        variant: Dictionary of hyperparameters
        log_dir: Directory for logs/checkpoints
        exp_name: Experiment name
    """
    # Lazy imports
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms

    # Import chester logger
    from chester import logger

    # Extract hyperparameters from variant
    model_type = variant.get('model', 'mlp')
    hidden_dim = variant.get('hidden_dim', 256)
    learning_rate = variant.get('learning_rate', 0.001)
    batch_size = variant.get('batch_size', 64)
    epochs = variant.get('epochs', 5)

    # Setup logging
    logger.configure(dir=log_dir, format_strs=['csv', 'stdout'])
    logger.log(f"Starting experiment: {exp_name}")
    logger.log(f"Model: {model_type}, LR: {learning_rate}, BS: {batch_size}, Hidden: {hidden_dim}")

    # Device
    device = get_device()
    logger.log(f"Using device: {device}")

    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download to a shared cache location
    data_dir = os.path.join(os.path.dirname(log_dir), 'mnist_data')
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Build model
    if model_type == 'cnn':
        model = build_cnn()
    else:
        model = build_mlp(hidden_dim=hidden_dim)

    model = model.to(device)
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Log metrics
        logger.logkv('epoch', epoch)
        logger.logkv('train_loss', round(train_loss, 4))
        logger.logkv('train_acc', round(train_acc, 4))
        logger.logkv('test_loss', round(test_loss, 4))
        logger.logkv('test_acc', round(test_acc, 4))
        logger.dumpkvs()

        if test_acc > best_acc:
            best_acc = test_acc
            # Save best model
            checkpoint_path = os.path.join(log_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, checkpoint_path)

    logger.log(f"Training complete! Best test accuracy: {best_acc:.4f}")
    logger.reset()

    return {'best_acc': best_acc}


def main():
    """Standalone execution for testing."""
    parser = argparse.ArgumentParser(description='MNIST Training')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'])
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='./data/mnist_test')
    parser.add_argument('--exp_name', type=str, default='mnist_standalone')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    variant = {
        'model': args.model,
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
    }

    run_training(variant, args.log_dir, args.exp_name)


if __name__ == '__main__':
    main()
