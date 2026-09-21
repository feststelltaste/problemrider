---
title: Self-Service-Entwicklerplattform
description: Dinge, für die Entwickler um Erlaubnis bitten oder warten
  müssen — Umgebungen, Zugriff, Deployments, Daten — in Fähigkeiten
  verwandeln, die sie selbst innerhalb von Leitplanken aufrufen können.
category:
- Operations
- Process
- Team
problems:
- approval-dependencies
- work-blocking
- development-disruption
- inefficient-development-environment
- operational-overhead
- tool-limitations
- increased-manual-work
- inefficient-processes
- testing-environment-fragility
- bottleneck-formation
- inadequate-test-infrastructure
- extended-cycle-times
- wasted-development-effort
layout: solution
lang: de
en_slug: self-service-developer-platform
related_solutions:
- slug: development-environment-optimization
  similarity: 0.7
- slug: development-workflow-automation
  similarity: 0.7
- slug: ci-cd-pipeline
  similarity: 0.65
- slug: delivery-performance-metrics
  similarity: 0.65
- slug: customization-under-version-control
  similarity: 0.65
- slug: blue-green-canary-deployments
  similarity: 0.65
---

## Description

Eine Self-Service-Entwicklerplattform verwandelt die Dinge, die ein Team derzeit anfragen muss — eine Testumgebung, Datenbankzugriff, eine Bereitstellung, einen Satz Testdaten, ein neues Service-Gerüst —, in Fähigkeiten, die das Team selbst aufrufen kann, innerhalb von Leitplanken, die die Richtlinien kodieren, die der Anfrageprozess durchgesetzt hat. Der Unterschied zum bloßen Gewähren breiten Zugangs sind die Leitplanken: Die Plattform erlaubt, was die Richtlinie zulässt, und verhindert, was sie nicht zulässt, sodass Genehmigung im Mechanismus verkörpert ist, statt von einer Person für jede Instanz durchgeführt zu werden. Dies adressiert den Befund, den Value Stream Mapping fast immer produziert: dass der Großteil der verstrichenen Zeit einer Änderung damit verbracht wird, auf jemand anderen zu warten. In Legacy-Organisationen sind die angesammelten Anfrageprozesse üblicherweise die größte einzelne Komponente der Lieferzeit und die am wenigsten mit der Codebasis verbundene.

## How to Apply ◆

> Jeder Anfrageprozess in einer Legacy-Organisation wurde eingeführt, um etwas Spezifisches zu verhindern; das Ziel ist, diese Verhinderung zu bewahren, während die Person aus dem Pfad entfernt wird.

- **Beginnen Sie bei den gemessenen Wartezeiten**, nicht bei einer Plattform-Produktvision. Worauf Entwickler am längsten warten, ist die erste zu bauende Fähigkeit. Eine Plattform um das technisch Interessante herum zu bauen produziert etwas Beeindruckendes, an dem niemand blockiert war.
- **Kodieren Sie die Richtlinie in der Leitplanke** statt sie zu entfernen. Wenn eine Datenbankzugriffsanfrage existierte, um uneingeschränkten Produktionszugang zu verhindern, gewährt das Self-Service-Äquivalent zeitlich begrenzten, protokollierten, schreibgeschützten Zugang zu anonymisierten Daten. Die Kontrolle überlebt; die Warteschlange nicht.
- Machen Sie **Umgebungserstellung zum ersten Ziel** in den meisten Fällen. Gemeinsam genutzte Integrationsumgebungen sind eine Warteschlange, eine Quelle von Interferenz und eine Ursache nicht reproduzierbarer Fehler. Ephemere Per-Branch-Umgebungen beseitigen alle drei, und Containerisierung macht dies üblicherweise erreichbar.
- **Bieten Sie Golden Paths, kein Toolkit.** Ein einziger, gut unterstützter Weg, einen neuen Dienst zu erstellen, mit bereits verdrahteten Logging, Monitoring, Deployment und Secrets, ist es, was die Plattform übernommen macht. Eine Sammlung von Bausteinen lässt jedes Team sein eigenes zusammenstellen, was sie ohnehin schon taten.
- **Halten Sie die Plattform optional und machen Sie sie zur einfachsten Option.** Verpflichtende Plattformen brüten Groll und Workarounds; Plattformen, die echt schneller sind als die Alternative, werden ohne jegliches Mandat übernommen.
- **Protokollieren Sie alles, was die Plattform tut.** Für Auditoren und Sicherheitsteams akzeptables Self-Service ist Self-Service, das einen besseren Audit-Trail produziert als der manuelle Prozess, den es ersetzt — was üblicherweise leicht ist, da manuelle Genehmigungen häufig in E-Mail aufgezeichnet werden.
- **Behandeln Sie die Plattform als Produkt** mit Nutzern, Feedback und einer Roadmap. Als interne Infrastrukturprojekte gebaute und dann übergebene Plattformen neigen dazu, die Probleme der Erbauer statt der Nutzer zu lösen.
- **Beziehen Sie Testdatenbereitstellung ein**, ein chronisch unterversorgtes Bedürfnis. Ein Entwickler, der auf Anfrage einen realistischen, anonymisierten Datensatz erstellen kann, ist von einer Wartezeit entblockt, die sonst in Tagen gemessen wird.
- **Bauen Sie nicht, was Sie übernehmen können.** Die Wartungslast einer maßgeschneiderten Plattform in einer Organisation ohne dediziertes Plattformteam übersteigt häufig die Verzögerung, die sie beseitigt hat.

## Tradeoffs ⇄

> Self-Service beseitigt die größten Warteschlangen in den meisten Lieferprozessen, erfordert aber echte Investition, ein Team, das sie besitzt, und Kontrollen, sorgfältig übersetzt statt verworfen.

**Vorteile:**

- Die dominante Komponente der Zykluszeit — das Warten auf jemand anderen — wird direkt reduziert, was keine Menge schnelleren Codierens erreicht.
- Interferenz zwischen Teams, die Umgebungen teilen, verschwindet, zusammen mit den nicht reproduzierbaren Fehlern und blockierter Arbeit, die sie verursacht.
- Konsistenz verbessert sich, weil der Golden Path überall denselben Logging-, Monitoring- und Deployment-Ansatz anwendet, statt dass jedes Team seinen eigenen erfindet.
- Audit-Trails verbessern sich typischerweise, da eine Plattform jede Aktion aufzeichnet, während ein manueller Genehmigungsprozess eine E-Mail aufzeichnet.
- Die Menschen, die zuvor Anfragen bearbeiteten, werden für Arbeit freigesetzt, die Urteilsvermögen statt Wiederholung erfordert.

**Kosten und Risiken:**

- Der Bau und die Pflege einer Plattform ist eine erhebliche laufende Investition, und sie braucht ein besitzendes Team, sonst verfällt sie zu ungewartetem Tooling, das jeder umgeht.
- Sorglos übersetzte Leitplanken entfernen eine Kontrolle statt sie zu automatisieren, und die Lücke wird während eines Vorfalls oder Audits entdeckt.
- Eine Plattform kann zu ihrem eigenen Engpass werden, wenn jedes neue Bedürfnis erfordert, dass das Plattformteam es implementiert.
- Golden Paths schränken Wahl ein, was der Punkt ist und auch von Teams verübelt wird, deren Anforderungen echt abweichen.
- In Legacy-Landschaften können viele Systeme überhaupt nicht auf eine moderne Plattform gebracht werden, was zwei Arbeitsweisen und den Overhead beider hinterlässt.

## How It Could Be

Die Value Stream Map eines Teams zeigte, dass von 31 Kalendertagen von der Anfrage bis zur Produktion sechs mit dem Warten auf die gemeinsam genutzte Integrationsumgebung verbracht wurden, um die vier Teams konkurrierten, und drei mit dem Warten auf einen Datenbankadministrator zur Bereitstellung von Testdaten. Ihr Plattformaufwand ignorierte bewusst die Bereitstellungsautomatisierung, die bereits angemessen war, und zielte genau auf diese zwei Wartezeiten. Ephemere Per-Branch-Umgebungen brauchten ein Quartal zum Bau auf ihrer bestehenden Container-Infrastruktur. Self-Service-Testdatenbereitstellung — ein Skript, das auf Anfrage einen anonymisierten 4.000-Datensatz-Extrakt produziert, mit zweiwöchigem Ablauf — brauchte drei Wochen. Die neun Tage Wartezeit fielen zusammen auf unter zwei Stunden. Keine Fähigkeit war technisch bemerkenswert; beide waren einfach nie jemandes Aufgabe gewesen.

Die Leitplanken-Übersetzung zählte mehr als die Automatisierung. Produktionsdatenbankzugriff hatte eine schriftliche Anfrage und die Genehmigung eines Managers erfordert, was ein bis drei Tage dauerte, und wurde mehrmals monatlich gebraucht, um Defekte zu diagnostizieren. Der Self-Service-Ersatz gewährte schreibgeschützten Zugang zu einer Produktions-Replika, beschränkt auf eine festgelegte Menge von Tabellen, zeitlich auf vier Stunden begrenzt, mit jeder protokollierten Abfrage und der gegen den anfragenden Entwickler aufgezeichneten Sitzung. Das Sicherheitsteam genehmigte es in einem Meeting, weil es strikt kontrollierter war als die vorherige Regelung — unter der Zugang, einmal gewährt, uneingeschränkt, unbegrenzt und unprotokolliert gewesen war. Die dreitägige Wartezeit wurde zu einer Self-Service-Aktion, die unter einer Minute dauerte, und die tatsächliche Sicherheitslage der Organisation verbesserte sich.
