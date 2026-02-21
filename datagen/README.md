# Synthetic Data Generation

Generate training videos using Cosmos Predict 2.5 from Stanford Shopping Center site photos and text prompts.

## Modes

- **Image2World** — Real site photos → animated curbside delivery scenarios
- **Text2World** — Text prompts → diverse scenario generation for edge cases

## Usage

```bash
# Generate from site photos
python generate_image2world.py --photos site_photos/ --prompts prompts/image2world_base.json --output ../data/synthetic_clips/

# Generate from text prompts
python generate_text2world.py --prompts prompts/text2world_base.json --output ../data/synthetic_clips/

# Augment base prompts with LLM (20 base → 150+ variations)
python augment_prompts.py --input prompts/text2world_base.json --output prompts/text2world_augmented.json --count 150
```

## Site Photos

Place Stanford Shopping Center photos in `site_photos/`. Prioritize:
1. Loading zones from multiple angles
2. Close-ups of parking/loading zone signage
3. Wide shots showing full curbside layout with pedestrian paths
4. Areas where delivery handoff would be staged
