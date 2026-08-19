---
title: Blue-Green-Deployment
description: Betrieb zweier paralleler Produktivumgebungen, um Ausfallzeiten zu minimieren.
category:
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/blue-green-deployment/
problems:
- deployment-risk
- large-risky-releases
- release-instability
- missing-rollback-strategy
- frequent-hotfixes-and-rollbacks
- release-anxiety
- fear-of-breaking-changes
- system-outages
- service-timeouts
layout: solution
lang: de
en_slug: blue-green-canary-deployments
related_solutions:
- slug: ci-cd-pipeline
  similarity: 0.85
- slug: continuous-deployment
  similarity: 0.8
- slug: strangler-fig-pattern
  similarity: 0.75
- slug: feature-flags
  similarity: 0.75
- slug: rollback-mechanisms
  similarity: 0.75
- slug: smoke-testing
  similarity: 0.75
---

## Description

Blue-Green-Deployment betreibt zwei identische Produktivumgebungen nebeneinander und sendet Live-Traffic jeweils nur an eine, sodass das Ausliefern einer neuen Version bedeutet, einen Router auf die bereits verifizierte andere Umgebung umzuschalten, statt die laufende an Ort und Stelle zu ändern. Dies verwandelt Rollback von einer langsamen, riskanten Umkehrung eines In-Place-Deployments in einen sofortigen Traffic-Wechsel zurück zu der Umgebung, die Momente zuvor Anfragen bediente — genau die Eigenschaft, die Legacy-Systeme mit einer Geschichte langer, zeremonieller Release-Fenster und schwieriger Rollbacks am meisten brauchen. Der Ansatz verdoppelt ungefähr die Infrastrukturkosten für die Dauer eines Deployments und erfordert Datenbankschemaänderungen, die über beide Versionen hinweg abwärtskompatibel bleiben, was oft die schwierigste Anpassung für Legacy-Systeme ist, deren Migrationspraktiken davon ausgehen, dass immer nur eine Version läuft.

## How to Apply ◆

> Blue-Green-Deployment adressiert eines der akutesten Risiken im Legacy-Systembetrieb — das folgenreiche, seltene Release, das Monate an Änderungen anhäuft und einen schwierigen Rollback fürchtet.

- Etablieren Sie von Anfang an zwei produktionsäquivalente Umgebungen. Für Legacy-Systeme, die oft auf physischen Servern oder festen Cloud-Instanzen laufen, bedeutet dies, eine zweite Umgebung bereitzustellen, die die erste spiegelt. Infrastructure-as-Code-Werkzeuge können beide Umgebungen aus gemeinsamen Vorlagen mit umgebungsspezifischen Parametern verwalten.
- Platzieren Sie eine Routing-Schicht — einen Load Balancer oder Reverse Proxy — vor beiden Umgebungen. Dies ist der Kontrollpunkt für den Traffic-Wechsel. Stellen Sie sicher, dass die Routing-Schicht sofortigen Umschalt ohne abgebrochene Verbindungen unterstützt, unter Nutzung von Connection Draining für alle lang laufenden Legacy-Transaktionen.
- Nutzen Sie die inaktive Umgebung als Vor-Produktions-Verifikationsstufe. Für Legacy-Systeme, bei denen sich Staging-Umgebungen historisch von der Produktion unterschieden haben, fängt die Ausführung der finalen Verifikation auf produktionsgerechter Infrastruktur die Fehlerklasse ab, die Staging nicht kann: Konfigurationsunterschiede, Ressourcenbeschränkungen und Integrationsverhalten, das sich nur im Produktionsmaßstab zeigt.
- Adressieren Sie Datenbankschemamigrationen explizit vor jedem Deployment. Legacy-Systeme haben typischerweise große, gemeinsam genutzte Datenbanken. Übernehmen Sie das Expand-and-Contract-Muster: Fügen Sie zuerst neue Spalten oder Tabellen hinzu, deployen Sie die neue Anwendungsversion, entfernen Sie dann die alten Strukturen erst, nachdem Rollback nicht mehr benötigt wird. Führen Sie nie eine Migration aus, die die Abwärtskompatibilität mit der vorherigen Anwendungsversion bricht.
- Automatisieren Sie die Deploy-Verify-Switch-Sequenz. Manuelle Blue-Green-Deployments in Legacy-Umgebungen häufen kleine prozedurale Variationen an, die schließlich Fehler verursachen. Die Sequenz zu skripten und den Traffic-Wechsel hinter automatisierten Health Checks zu gaten eliminiert menschlichen Fehler aus dem kritischsten Schritt.
- Üben Sie Rollback regelmäßig, nicht nur wenn etwas schiefgeht. In Legacy-Umgebungen stellen sich Rollback-Prozeduren, die nie ausgeführt werden, oft als nicht funktionierend heraus, wenn sie am meisten gebraucht werden. Planen Sie periodische Rollback-Übungen als Teil der Routine des Teams.
- Wärmen Sie die inaktive Umgebung vor dem Traffic-Wechsel auf. Legacy-Systeme, die auf der JVM, dem .NET-Runtime oder ähnlichen Plattformen mit JIT-Kompilierung gebaut sind, brauchen Warm-up-Traffic, um Steady-State-Performance zu erreichen. Ebenso müssen Anwendungsebenen-Caches vor dem Wechsel befüllt werden, um einen Cache-Miss-Sturm zu vermeiden, der die Datenbank im Moment der Umschaltung überwältigt.
- Überwachen Sie Fehlerraten, Latenz und geschäftsebene Indikatoren genau für die ersten dreißig bis sechzig Minuten nach dem Wechsel. Für Legacy-Systeme mit komplexen Transaktionsabläufen könnten Probleme nicht sofort auftauchen, sondern erst, während Batch-Prozesse laufen oder seltenere Codepfade ausgeführt werden.

## Tradeoffs ⇄

> Blue-Green-Deployment verringert das Risiko jedes einzelnen Releases dramatisch, aber die Pflege zweier Produktivumgebungen fügt Kosten und operative Komplexität hinzu, die gegen die Release-Häufigkeit und Kritikalität des Legacy-Systems gerechtfertigt werden müssen.

**Vorteile:**

- Rollback wird sofortig und sicher: Wenn die neue Version nach dem Traffic-Wechsel fehlschlägt, stellt das Zurückleiten des Traffics zur vorherigen Umgebung den Service in Sekunden wieder her, ohne die Notwendigkeit, einen komplexen In-Place-Deployment-Prozess umzukehren.
- Die inaktive Umgebung bietet produktionsgerechte Vor-Release-Verifikation, die Staging-Umgebungen — die sich in Legacy-Systemen oft erheblich von der Produktion unterscheiden — nicht replizieren können. Dies fängt die Fehler ab, die historisch die schädlichsten Vorfälle verursacht haben.
- Deployment-Angst verringert sich, weil jedes Release nicht mehr in dem Moment unumkehrbar ist, in dem es live geht. Teams, die zuvor Änderungen in großen, seltenen Releases anhäuften (um die Häufigkeit hochriskanter Deployments zu minimieren), können kleinere Batches häufiger ausliefern, was das kumulative Risiko pro Release verringert.
- Deployment und Release sind zeitlich getrennt. Code kann in die inaktive Umgebung deployt, verifiziert und gehalten werden, bis das Geschäft zum Release bereit ist — ohne den Druck eines engen Wartungsfensters und ohne dass Nutzer unverifiziertem Code ausgesetzt sind.
- Für Legacy-Systeme mit Verfügbarkeitsanforderungen eliminiert oder verringert Blue-Green dramatisch deployment-bedingte Ausfallzeit und beseitigt die Notwendigkeit von Wartungsfenstern und der damit verbundenen Geschäftsstörung.

**Kosten und Risiken:**

- Der Betrieb zweier vollständiger Produktivumgebungen verdoppelt ungefähr die Infrastrukturkosten für die Dauer der Deployment-Periode. Für Legacy-Systeme, die auf teurer On-Premises-Hardware laufen, könnten diese Kosten prohibitiv sein. Cloud-basierte Umgebungen können dies mildern, indem die inaktive Umgebung zwischen Deployments herunterskaliert wird.
- Legacy-Systeme mit großen, gemeinsam genutzten Datenbanken präsentieren die schwierigste Herausforderung. Datenbankschemaänderungen, die nicht abwärtskompatibel sind, verhindern sicheren Rollback, selbst wenn die Anwendungsinfrastruktur ihn unterstützt. Teams entdecken oft mitten in der Übernahme, dass ihre bestehenden Migrationspraktiken mit Blue-Green-Deployment inkompatibel sind.
- Konfigurations-Drift zwischen der Blue- und Green-Umgebung ist ein anhaltendes Risiko. Ohne Infrastructure as Code, das beide Umgebungen aus gemeinsamen Definitionen verwaltet, häufen sich Unterschiede über die Zeit an und verursachen Fehler während des Wechsels, die während der Verifikation der inaktiven Umgebung nicht auftauchen.
- Legacy-Systeme mit lang laufenden Transaktionen, persistenten Verbindungen oder zustandsbehafteten Protokollen erschweren den Traffic-Wechsel. Das elegante Ablassen aktiver Sitzungen während des Routing-Wechsels erfordert Verständnis des Verbindungsmodells des Systems — Wissen, das möglicherweise nicht dokumentiert oder auch nur gut verstanden ist.
- Die erforderliche organisatorische Veränderung ist erheblich. Legacy-Teams, die an seltene, zeremonielle Deployments gewöhnt sind, müssen sich an ein anderes operatives Modell anpassen. Die Prozesse, Rollen und das Tooling zur gleichzeitigen Verwaltung zweier Umgebungen erfordern Investition zur Etablierung.

## How It Could Be

> Legacy-Systeme profitieren am meisten von Blue-Green-Deployment, wenn ihre Release-Geschichte von schmerzhaften Rollbacks, langwierigen Ausfallfenstern oder einer Kultur der Vermeidung von Releases dominiert wird, weil jedes zu riskant ist.

Eine regionale Bank, die ein Kernbanksystem On-Premises betrieb, hatte Releases auf vierteljährliche Ereignisse begrenzt, jedes ein Freitagabend-Wartungsfenster erfordernd, während dessen das System für Filialpersonal und Geldautomaten vollständig nicht verfügbar war. Der Deployment-Prozess beinhaltete manuelles Stoppen von Services, Deployen neuer Binärdateien, Ausführen von Datenbankmigrationsskripten und Neustarten — eine Sequenz, die zwischen zwei und fünf Stunden dauerte und eine dokumentierte Rollback-Rate von ungefähr einem von fünf Deployments hatte. Nach dem Aufbau einer zweiten Umgebung unter Nutzung derselben Hardware-Spezifikationen und der Einführung eines F5-Load-Balancers als Routing-Schicht wechselte das Team zu monatlichen Releases ohne Wartungsfenster. Beim ersten Mal, als sie Rollback unter dem neuen Modell ausführten, dauerte es acht Sekunden. Das vierteljährliche Ausfallfenster, das die Bank messbare Einnahmen gekostet und umfangreiche Kundenkommunikation jedes Quartal erfordert hatte, wurde vollständig eliminiert.

Das Paketverfolgungssystem eines Logistikunternehmens handhabte mehrere Millionen Statusupdates pro Tag und konnte nicht mehr als ein paar Minuten Ausfallzeit während der Geschäftsstunden tolerieren. Ihr vorheriger Deployment-Ansatz erforderte das Deployen während eines zweistündigen Fensters am Sonntagmorgen, was mit internationaler Sendungsverarbeitung kollidierte und immer noch gelegentliche Vorfälle verursachte, die weit über das Fenster hinausreichten. Durch die Einführung von Blue-Green-Deployment auf ihrer Cloud-Infrastruktur konnten sie Verfolgungsservice-Updates jederzeit deployen, sie gegen Produktions-Traffic auf der inaktiven Umgebung verifizieren und ohne messbare Serviceunterbrechung wechseln. Innerhalb eines Jahres hatten sie ihre Release-Häufigkeit von wöchentlichen Sonntagsfenstern auf mehrmals pro Woche erhöht, wobei jedes einzelne Release weit weniger Risiko trug, weil der Änderungssatz kleiner war.

Das Check-in-System einer Fluggesellschaft hatte eine Geschichte fehlgeschlagener Deployments, die Notfall-Rollbacks unter Druck erforderten, oft während Reisespitzenzeiten, wenn das Timing eines Releases falsch eingeschätzt worden war. Jeder Rollback erforderte ein Betriebsteam, um das Deployment manuell umzukehren, während Ingenieure den Fehler diagnostizierten — ein Prozess, der zwischen zwanzig Minuten und zwei Stunden dauerte, abhängig von der Art des Problems. Die Einführung von Blue-Green-Deployment, kombiniert mit automatisierten Smoke-Tests, die Check-in-Abläufe gegen die inaktive Umgebung vor dem Traffic-Wechsel verifizierten, eliminierte ungeplante Rollbacks vollständig über die folgenden achtzehn Monate. Die Tests fingen drei Releases ab, die zuvor Produktion in einem defekten Zustand erreicht hätten; alle drei wurden von der inaktiven Umgebung zurückgesetzt, bevor irgendein Nutzer betroffen war.
