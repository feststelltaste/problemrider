---
title: Stammdaten-Stewardship
description: Jedem gemeinsam genutzten Referenzobjekt einen verantwortlichen
  Steward, einen definierten Erstellungsprozess und einen gemessenen
  Qualitätsstandard geben, sodass abteilungsübergreifende Daten einen Eigentümer
  haben.
category:
- Database
- Management
- Business
problems:
- master-data-ownership-gaps
- shadow-systems
- poor-interfaces-between-applications
- data-migration-complexities
- duplicated-effort
- lack-of-ownership-and-accountability
- custom-report-sprawl
- increased-manual-work
- inconsistent-execution
- data-migration-integrity-issues
- entity-attribute-value-overuse
- system-integration-blindness
layout: solution
lang: de
en_slug: master-data-stewardship
related_solutions:
- slug: clear-ownership-model
  similarity: 0.6
- slug: clear-roles-and-ownership
  similarity: 0.6
- slug: data-quality-checks
  similarity: 0.6
- slug: attribute-usage-analysis
  similarity: 0.6
- slug: product-owner
  similarity: 0.6
- slug: role-model-rationalization
  similarity: 0.55
---

## Description

Stammdaten-Stewardship weist einer Person die Rechenschaftspflicht pro gemeinsam genutztem Referenzobjekt zu — Kunde, Lieferant, Produkt, Kostenstelle —, die für die Qualität dieses Objekts als Ganzes verantwortlich ist, unterstützt durch einen definierten Erstellungsprozess und einen gemessenen Standard. Es adressiert eine strukturelle Lücke statt einer technischen. Gemeinsam genutzte Daten werden von mehreren Abteilungen gepflegt, von denen jede sich um die Felder kümmert, die sie nutzt, und keine die vollen Kosten erfährt, wenn das Objekt falsch ist. Unter dieser Regelung verschlechtert sich die Qualität zuverlässig, und die Verschlechterung tritt als Symptome an anderen Stellen zutage — ausfallende Schnittstellen, widersprüchliche Berichte, doppelte Zahlungen —, die den meldenden Systemen statt den Daten zugeschrieben werden. Ein Steward ist der Mechanismus, durch den ein Objekt, das Organisationsgrenzen überschreitet, einen einzigen Verantwortungspunkt erwirbt.

## How to Apply ◆

> Daten, die Abteilungen überschreiten, haben in den meisten Organisationen keinen Eigentümer, weil Eigentümerschaft entlang derselben funktionalen Linien zugewiesen wird, die die Daten überschreiten.

- **Benennen Sie einen Steward pro Objekt**, eine Person statt eines Ausschusses oder einer Abteilung. Der Steward muss die Daten nicht pflegen; er muss dafür verantwortlich sein, ob sie korrekt sind, und der Adressat für Fragen dazu sein.
- **Definieren Sie den Erstellungsprozess**, einschließlich eines verpflichtenden Suchschritts, bevor ein neuer Datensatz erstellt werden darf. Duplikate entstehen, weil Erstellen schneller ist als Suchen, und der Prozess ist es, was diese Rechnung ändert.
- **Vereinbaren Sie den Qualitätsstandard pro Objekt**: welche Felder in der Praxis verpflichtend sind, unabhängig davon, welche Abteilung sie braucht, was die Konventionen sind und was ein Duplikat ausmacht. Abteilungen werden sich uneinig sein, und dies zu klären ist die erste Aufgabe des Stewards.
- **Messen Sie Qualität kontinuierlich** — Duplikatrate, Vollständigkeit, Konventionsverstöße — und senden Sie den Bericht an den Steward statt an eine Verteilerliste. Ein Maß ohne Adressaten produziert keine Handlung.
- **Trennen Sie Erstellungsberechtigung von Bearbeitungsberechtigung.** Zu beschränken, wer Datensätze erstellen darf, während breite Bearbeitung der Felder erlaubt bleibt, die eine Abteilung besitzt, adressiert Duplikation, ohne die Daten schwer pflegbar zu machen.
- **Arbeiten Sie den bestehenden Rückstand als begrenztes Arbeitsstück** mit einem Eigentümer und einem Ende ab, statt als Dauerzustand. Duplikatauflösung ist endlich, sobald neue Duplikaterstellung gestoppt wurde.
- **Geben Sie nachgelagerten Konsumenten einen Weg, Probleme** an den Steward zu melden. Konsumierende Systeme erkennen Qualitätsprobleme zuerst und haben meist nirgendwo, wohin sie sie schicken können, sodass sie sie stattdessen still umgehen.
- **Adressieren Sie die Ursachen statt der Einzelfälle.** Ein wiederkehrendes Qualitätsproblem in einem Feld deutet meist auf eine Prozess- oder Validierungslücke hin, und einzelne Datensätze für immer zu korrigieren ist die Alternative dazu, sie zu finden.
- **Überprüfen Sie den Standard, wenn sich das Geschäft ändert.** Neue Märkte, neue Rechtsträger und neue Produkttypen ändern, was korrekt bedeutet, und ein einmal gesetzter Standard driftet aus der Nützlichkeit heraus.

## Tradeoffs ⇄

> Stewardship behebt die Rechenschaftslücke, die gemeinsam genutzte Daten verkommen lässt, auf Kosten einer Rolle, die jemand ausfüllen muss, und Prozessreibung am Punkt der Erstellung.

**Vorteile:**

- Gemeinsam genutzte Daten erwerben einen einzigen Rechenschaftspunkt, was die strukturelle Bedingung ist, von der ihre Qualität abhängt.
- Die Duplikaterstellung fällt stark, sobald Suche verpflichtend und Erstellungsberechtigung beschränkt ist, was die größte einzelne Quelle der Verschlechterung ist.
- Nachgelagerte Ausfälle nehmen ab, da Schnittstellen und Berichte von der Qualität genau dieser Daten abhängen.
- Migrationen werden erheblich günstiger, weil die Auflösung angesammelter Duplikate und Inkonsistenzen meist der größte Bestandteil des Migrationsaufwands ist.
- Widersprüchliches Reporting wird verringert, weil die Definitionen, auf die sich die Berichte stützen, vereinbart und besessen statt angenommen werden.

**Kosten und Risiken:**

- Die Steward-Rolle ist echte Arbeit, die ressourciert werden muss, und sie wird häufig zum bestehenden Job von jemandem hinzugefügt, wo sie dann nicht erledigt wird.
- Die Beschränkung der Erstellungsberechtigung fügt Reibung in dem Moment hinzu, in dem ein Nutzer versucht, eine Aufgabe abzuschließen, was Widerstand und Workarounds erzeugt.
- Stewardship überschreitet Abteilungsgrenzen und erfordert daher Autorität, die der Steward möglicherweise nicht hat, was die Rolle frustrierend und schwer zu besetzen macht.
- Qualitätsstandards können bürokratisch werden und Felder verlangen, die niemand nutzt, weil ein Ausschuss dachte, sie könnten zählen.
- Der bestehende Rückstand schlechter Daten kann groß genug sein, um demoralisierend zu wirken, und seine Bereinigung liefert keine sichtbare Fähigkeit.

## How It Could Be

Der Lieferantenstamm eines Fertigungsunternehmens enthielt etwa 6.800 wahrscheinliche Duplikate in 41.000 Datensätzen, entstanden weil Einkauf und Finanzen jeweils Datensätze hinzufügten, um ihre eigene Aufgabe abzuschließen, und keine gründlich suchte. Der Eingriff war organisatorisch statt technisch: ein benannter Steward für Lieferantendaten, ein Erstellungsprozess, der eine Suche mit protokolliertem Ergebnis erforderte, Erstellungsberechtigung beschränkt auf eine kleine Gruppe und ein monatlicher Duplikatbericht, persönlich an den Steward gesendet. Neue Duplikaterstellung fiel innerhalb von zwei Quartalen um etwa neunzig Prozent. Der bestehende Rückstand wurde dann als begrenztes Projekt statt als Umgebungszustand bearbeitet, was etwa fünf Monate dauerte.

Der nachgelagerte Effekt war größer als der Datenqualitätseffekt. Zwei Konsequenzen, die anderswo zugeschrieben worden waren, lösten sich von selbst: Die Ausgabenanalyse hatte die Konzentration bei einzelnen Lieferanten unterschätzt, was im Vorjahr eine Verhandlungsposition geschwächt hatte, und ein Zahlungslauf hatte zweimal doppelte Zahlungen unter zwei Datensätzen für denselben Lieferanten ausgelöst. Keines war als Stammdatenproblem erkannt worden — das erste war als Reporting-Defizit behandelt worden und das zweite als Zahlungsprozessfehler. Der nützlichste Beitrag des Stewards im ersten Jahr war, ein Adressat zu sein: drei Abteilungen, die jahrelang still Lieferantendatenprobleme umgangen hatten, hatten endlich einen Ort, wohin sie diese schicken konnten.
