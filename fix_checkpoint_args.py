from pathlib import Path

path = Path("backend/ml/training/train.py")
text = path.read_text(encoding="utf-8")

changes = 0

# ------------------------------------------------------------
# Add best_model_state to train_one_epoch signature if missing
# ------------------------------------------------------------

signature_old = """    start_batch,
    best_f1,
    epochs_without_improvement,
):"""

signature_new = """    start_batch,
    best_f1,
    best_model_state,
    epochs_without_improvement,
):"""

if signature_old in text:
    text = text.replace(
        signature_old,
        signature_new,
        1,
    )
    changes += 1


# ------------------------------------------------------------
# Add best_model_state to periodic checkpoint call if missing
# ------------------------------------------------------------

checkpoint_old = """                    scaler=scaler,
                    best_f1=best_f1,
                    epochs_without_improvement="""

checkpoint_new = """                    scaler=scaler,
                    best_f1=best_f1,
                    best_model_state=best_model_state,
                    epochs_without_improvement="""

if checkpoint_old in text:
    text = text.replace(
        checkpoint_old,
        checkpoint_new,
        1,
    )
    changes += 1


path.write_text(
    text,
    encoding="utf-8",
)

print(
    f"SUCCESS: Applied {changes} missing checkpoint fix(es)."
)
