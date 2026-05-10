def predict_batch(events_batch):
    # events_batch 遵循统一登录Schema
    df = preprocess_batch(events_batch)
    g = build_spatiotemporal_graph(df)
    seq = build_sequences(df)
    model.load_state_dict(torch.load("astgnn_login.pth"))
    model.eval()
    with torch.no_grad():
        score = torch.softmax(model(g, seq), dim=1)[:,1]
    return score.numpy()