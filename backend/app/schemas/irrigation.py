from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import date

ALLOWED_CROPS = {"Arecanut", "Coconut", "Pepper"}
ALLOWED_GROWTH_STAGES = {"Vegetative", "Flowering", "Fruiting", "Dormant"}

class IrrigationDayPlan(BaseModel):
    date: date
    crop: str
    decision: str
    water_liters: float
    reason: str
    soil_moisture_forecast: float

class IrrigationPredictRequest(BaseModel):
    soil_moisture: float = Field(..., ge=0.0, le=1.0)    # 0.0 - 1.0
    crop_types: list[str] = Field(..., min_length=1)     # ["Arecanut", "Coconut", "Pepper"]
    growth_stages: dict[str, str] = Field(default_factory=dict)     # {"Arecanut": "Vegetative", ...}
    model: str = "LSTM"
    num_plants: dict[str, int] = Field(default_factory=dict)   # {"Arecanut": 50, "Coconut": 30}

    @field_validator("crop_types")
    @classmethod
    def validate_crop_types(cls, crops: list[str]) -> list[str]:
        normalized = [c.strip() for c in crops if c and c.strip()]
        if not normalized:
            raise ValueError("crop_types must not be empty")
        invalid = [c for c in normalized if c not in ALLOWED_CROPS]
        if invalid:
            raise ValueError(f"Unsupported crop(s): {invalid}. Allowed: {sorted(ALLOWED_CROPS)}")
        return normalized

    @model_validator(mode="after")
    def fill_and_validate_maps(self):
        for crop in self.crop_types:
            stage = self.growth_stages.get(crop, "Vegetative")
            if stage not in ALLOWED_GROWTH_STAGES:
                raise ValueError(
                    f"Invalid growth stage '{stage}' for crop '{crop}'. Allowed: {sorted(ALLOWED_GROWTH_STAGES)}"
                )
            self.growth_stages[crop] = stage

            plants = self.num_plants.get(crop, 1)
            if plants < 1:
                raise ValueError(f"num_plants for '{crop}' must be >= 1")
            self.num_plants[crop] = plants

        # Reject extraneous keys to avoid silent typos
        extra_stage_keys = set(self.growth_stages.keys()) - set(self.crop_types)
        if extra_stage_keys:
            raise ValueError(f"growth_stages contains crops not in crop_types: {sorted(extra_stage_keys)}")

        extra_plant_keys = set(self.num_plants.keys()) - set(self.crop_types)
        if extra_plant_keys:
            raise ValueError(f"num_plants contains crops not in crop_types: {sorted(extra_plant_keys)}")

        return self

class IrrigationPredictResponse(BaseModel):
    plan: list[IrrigationDayPlan]
    total_water_liters: dict[str, float]  # per crop, 14-day total
    model_used: str
