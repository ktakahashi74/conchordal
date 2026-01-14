use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::Path;

use conchordal::life::control::AgentPatch;
use conchordal::life::scenario::{SpawnMethod, TimbreGenotype};
use schemars::{Schema, schema_for};
use serde_json::{Map, Value};

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = Path::new("docs/schemas");
    fs::create_dir_all(out_dir)?;

    write_schema_markdown("agent_patch", &schema_for!(AgentPatch), out_dir)?;
    write_schema_markdown("spawn_method", &schema_for!(SpawnMethod), out_dir)?;
    write_schema_markdown("timbre_genotype", &schema_for!(TimbreGenotype), out_dir)?;

    Ok(())
}

fn write_schema_markdown(name: &str, root: &Schema, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let mut output = String::new();
    append_schema_section(&mut output, "Root", root);

    let defs = schema_definitions(root);

    for (def_name, schema) in defs.iter() {
        output.push('\n');
        output.push_str(&format!("## {def_name}\n\n"));
        append_schema_section(&mut output, def_name, schema);
    }

    let path = out_dir.join(format!("{name}.md"));
    fs::write(path, output)?;
    Ok(())
}

fn append_schema_section(output: &mut String, label: &str, schema: &Schema) {
    match schema.as_bool() {
        Some(_) => {
            output.push_str(&format!("{label}: any\n"));
        }
        None => {
            let Some(obj) = schema.as_object() else {
                output.push_str(&format!("{label}: any\n"));
                return;
            };
            if let Some(table) = schema_object_table(obj, Some(label)) {
                output.push_str(&table);
                return;
            }

            if append_variant_tables(output, obj) {
                return;
            }

            let summary = schema_object_summary(obj);
            output.push_str(&format!("{summary}\n"));
        }
    }
}

fn schema_object_table(obj: &Map<String, Value>, label: Option<&str>) -> Option<String> {
    let properties = obj
        .get("properties")
        .and_then(Value::as_object)
        .filter(|props| !props.is_empty())?;

    let mut out = String::new();
    out.push_str("| Field | Type | Description |\n");
    out.push_str("| --- | --- | --- |\n");

    let keys = ordered_property_keys(properties, label);

    for key in keys {
        let schema = &properties[key];
        let type_name = schema_type_name(schema);
        let description = schema_description(schema);
        out.push_str(&format!("| {key} | {type_name} | {description} |\n"));
    }

    out.push('\n');
    Some(out)
}

fn append_variant_tables(output: &mut String, obj: &Map<String, Value>) -> bool {
    let variants = obj
        .get("oneOf")
        .and_then(Value::as_array)
        .filter(|list| !list.is_empty())
        .or_else(|| {
            obj.get("anyOf")
                .and_then(Value::as_array)
                .filter(|list| !list.is_empty())
        });
    let Some(variants) = variants else {
        return false;
    };

    if let Some((values, complete)) = union_string_values(variants) {
        let joined = values.join(", ");
        output.push_str(&format!("Enum values: {joined}\n\n"));
        if !complete {
            output.push_str("Schema details are complex; see JSON schema.\n\n");
        }
        return true;
    }

    let mut emitted = false;
    for (idx, schema) in variants.iter().enumerate() {
        let name = variant_name(schema).unwrap_or_else(|| format!("Variant {idx}"));
        output.push_str(&format!("### {name}\n\n"));
        if let Some(obj) = schema.as_object() {
            if let Some(table) = schema_object_table(obj, None) {
                output.push_str(&table);
                emitted = true;
                continue;
            }
        }

        let summary = schema_summary(schema);
        output.push_str(&format!("{summary}\n\n"));
        emitted = true;
    }

    emitted
}

fn union_string_values(variants: &[Value]) -> Option<(Vec<String>, bool)> {
    let mut values = Vec::new();
    let mut complete = true;
    for variant in variants {
        if let Some(mut next) = schema_enum_values(variant) {
            values.append(&mut next);
        } else {
            complete = false;
        }
    }
    if values.is_empty() {
        return None;
    }
    values.sort();
    values.dedup();
    Some((values, complete))
}

fn schema_enum_values(schema: &Value) -> Option<Vec<String>> {
    let obj = schema.as_object()?;
    if let Some(value) = obj.get("const").and_then(Value::as_str) {
        return Some(vec![value.to_string()]);
    }
    let values = obj.get("enum")?.as_array()?;
    let mut out = Vec::new();
    for value in values {
        let value = value.as_str()?;
        out.push(value.to_string());
    }
    if out.is_empty() { None } else { Some(out) }
}

fn schema_object_summary(obj: &Map<String, Value>) -> String {
    if let Some(values) = obj.get("enum").and_then(Value::as_array) {
        let items = values
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        if !items.is_empty() {
            return format!("Enum values: {items}");
        }
    }

    if let Some(reference) = obj.get("$ref").and_then(Value::as_str) {
        return format!("Reference: {reference}");
    }

    "Schema details are complex; see JSON schema.".to_string()
}

fn schema_summary(schema: &Value) -> String {
    match schema {
        Value::Bool(_) => "Schema details are complex; see JSON schema.".to_string(),
        Value::Object(obj) => schema_object_summary(obj),
        _ => "Schema details are complex; see JSON schema.".to_string(),
    }
}

fn ordered_property_keys<'a>(
    properties: &'a Map<String, Value>,
    label: Option<&str>,
) -> Vec<&'a String> {
    let mut keys: Vec<_> = properties.keys().collect();
    let pitch_order = [
        "mode",
        "freq",
        "range_oct",
        "gravity",
        "exploration",
        "persistence",
    ];
    if matches!(label, Some("PitchControl") | Some("PitchPatch")) {
        let mut ordered = Vec::new();
        let mut seen = HashSet::new();
        for name in pitch_order {
            if let Some(key) = keys.iter().find(|k| k.as_str() == name).copied() {
                ordered.push(key);
                seen.insert(key.as_str());
            }
        }
        let mut rest: Vec<_> = keys
            .into_iter()
            .filter(|k| !seen.contains(k.as_str()))
            .collect();
        rest.sort();
        ordered.extend(rest);
        return ordered;
    }
    keys.sort();
    keys
}

fn schema_type_name(schema: &Value) -> String {
    match schema {
        Value::Bool(_) => "any".to_string(),
        Value::Object(obj) => {
            if let Some(reference) = obj.get("$ref").and_then(Value::as_str) {
                return reference_name(reference);
            }

            if let Some(instance_type) = obj.get("type") {
                return match instance_type {
                    Value::String(t) => instance_type_name(t, obj),
                    Value::Array(list) => {
                        let mut names = list
                            .iter()
                            .filter_map(Value::as_str)
                            .map(|t| instance_type_name(t, obj))
                            .collect::<Vec<_>>();
                        names.sort();
                        names.join(" or ")
                    }
                    _ => "object".to_string(),
                };
            }

            if obj.get("oneOf").is_some() || obj.get("anyOf").is_some() {
                return "enum/union".to_string();
            }

            "object".to_string()
        }
        _ => "any".to_string(),
    }
}

fn instance_type_name(instance_type: &str, obj: &Map<String, Value>) -> String {
    match instance_type {
        "array" => array_item_type(obj).unwrap_or_else(|| "array".to_string()),
        "boolean" => "bool".to_string(),
        "integer" => "integer".to_string(),
        "null" => "null".to_string(),
        "number" => "number".to_string(),
        "object" => "object".to_string(),
        "string" => "string".to_string(),
        _ => "object".to_string(),
    }
}

fn array_item_type(obj: &Map<String, Value>) -> Option<String> {
    let items = obj.get("items")?;
    let item_schema = match items {
        Value::Object(_) | Value::Bool(_) => items,
        Value::Array(list) => list.first()?,
        _ => return None,
    };
    Some(format!("{}[]", schema_type_name(item_schema)))
}

fn schema_description(schema: &Value) -> String {
    match schema {
        Value::Bool(_) => String::new(),
        Value::Object(obj) => obj
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

fn reference_name(reference: &str) -> String {
    reference
        .rsplit('/')
        .next()
        .map(|s| s.to_string())
        .unwrap_or_else(|| reference.to_string())
}

fn variant_name(schema: &Value) -> Option<String> {
    let obj = schema.as_object()?;
    if let Some(reference) = obj.get("$ref").and_then(Value::as_str) {
        return Some(reference_name(reference));
    }
    if let Some(title) = obj.get("title").and_then(Value::as_str) {
        return Some(title.to_string());
    }
    let props = obj.get("properties")?.as_object()?;
    let prop = props
        .get("mode")
        .or_else(|| props.get("core"))
        .or_else(|| props.get("type"))?;
    let prop_obj = prop.as_object()?;
    if let Some(value) = prop_obj.get("const").and_then(Value::as_str) {
        return Some(value.to_string());
    }
    let values = prop_obj.get("enum")?.as_array()?;
    values
        .first()
        .and_then(Value::as_str)
        .map(|v| v.to_string())
}

fn schema_definitions(root: &Schema) -> BTreeMap<String, Schema> {
    let Some(obj) = root.as_object() else {
        return BTreeMap::new();
    };
    let defs = obj
        .get("$defs")
        .or_else(|| obj.get("definitions"))
        .and_then(Value::as_object);
    let Some(defs) = defs else {
        return BTreeMap::new();
    };
    defs.iter()
        .filter_map(|(k, v)| {
            let schema = Schema::try_from(v.clone()).ok()?;
            Some((k.to_string(), schema))
        })
        .collect()
}
